import aiofiles
import argparse
import asyncio
import collections
import hashlib
import itertools
import json
import math
import numpy
import os
import platformdirs
import psutil
import scipy
import sklearn
import sqlite3
import textual
import textual.app
import textual.widgets
import tenacity
import tiktoken

from asyncio import Semaphore
from dataclasses import dataclass
from dulwich.errors import NotGitRepository
from dulwich.repo import Repo
from itertools import batched, chain
from json import JSONDecodeError
from numpy import float32
from numpy.typing import NDArray
from pathlib import PurePath
from pydantic import BaseModel
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry
from sklearn.neighbors import NearestNeighbors
from tiktoken import Encoding
from typing import Iterable
from tqdm.asyncio import tqdm_asyncio

max_clusters = 20

@dataclass(frozen = True)
class Facets:
    embedding_client: AsyncOpenAI
    completion_client: AsyncOpenAI
    embedding_model: str
    completion_model: str
    embedding_encoding: Encoding
    completion_encoding: Encoding
    semaphore: Semaphore
    cache_directory: str

def initialize(
    embedding_model : str,
    completion_model: str,
    embedding_base_url : str | None = None,
    completion_base_url: str | None = None,
    embedding_encoding_name : str | None = None,
    completion_encoding_name: str | None = None,
) -> Facets:
    embedding_client  = AsyncOpenAI(base_url = embedding_base_url )
    completion_client = AsyncOpenAI(base_url = completion_base_url)

    embedding_model  = embedding_model
    completion_model = completion_model

    embedding_encoding = tiktoken.encoding_for_model(embedding_model)

    def to_encoding(encoding_name: str | None, model: str) -> Encoding:
        if encoding_name is None:
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                return tiktoken.get_encoding("o200k_base")
        else:
            return tiktoken.get_encoding(encoding_name)

    completion_encoding = to_encoding(completion_encoding_name, completion_model)
    embedding_encoding  = to_encoding(embedding_encoding_name , embedding_model )

    cache_directory = platformdirs.user_cache_dir(
        appname = "semantic-navigator",
        ensure_exists = True
    )

    # Half the smallest `ulimit` value commonly found in the wild (1024 on
    # Linux)
    available_descriptors = ensure_open_descriptors()

    semaphore = asyncio.Semaphore(available_descriptors)

    return Facets(
        embedding_client = embedding_client,
        completion_client = completion_client,
        embedding_model = embedding_model,
        completion_model = completion_model,
        embedding_encoding = embedding_encoding,
        completion_encoding = completion_encoding,
        semaphore = semaphore,
        cache_directory = cache_directory
    )

@dataclass(frozen = True)
class Embed:
    entry: str
    content: str
    embedding: NDArray[float32]

@dataclass(frozen = True)
class EmbedInput:
    entry: str
    content: str
    file_hash: str
    input_hash: str
    chunk_index: int

@dataclass(frozen = True)
class Cluster:
    embeds: list[Embed]

max_tokens_per_embed = 8192

max_tokens_per_batch_embed = 300000

# Increase the soft limit on the number of open file descriptors and return the
# available number of descriptors our code should use
def ensure_open_descriptors() -> int:
    # Use the current number of file descriptors in use to estimate how many
    # file descriptors to reserve
    num_fds = psutil.Process(os.getpid()).num_fds()

    reserve = 3 * num_fds

    if os.name == "posix":
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        desired = max(soft, hard - reserve)

        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))

        return hard
    else:
        # A quick search shows that the default limit (imposed by stdio) is 512
        # on Windows, which is probably a safe value to use for all non-POSIX
        # platforms
        return 512 - reserve

token_confirmation_threshold = 100

cache_schema_version = 1

def _cache_database_path() -> str:
    cache_home = os.environ.get("XDG_CACHE_HOME")

    if not cache_home:
        cache_home = os.path.join(os.path.expanduser("~"), ".cache")

    cache_directory = os.path.join(cache_home, "semantic-navigator")

    os.makedirs(cache_directory, exist_ok = True)

    return os.path.join(cache_directory, "cache.sqlite3")

def _open_cache_connection() -> sqlite3.Connection | None:
    try:
        connection = sqlite3.connect(_cache_database_path())
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                embedding_model TEXT NOT NULL,
                input_hash      TEXT NOT NULL,
                file_hash       TEXT NOT NULL,
                path            TEXT NOT NULL,
                chunk_index     INTEGER NOT NULL,
                schema_version  INTEGER NOT NULL,
                embedding       BLOB NOT NULL,
                created_at      INTEGER NOT NULL DEFAULT (unixepoch()),
                PRIMARY KEY (embedding_model, input_hash)
            )
            """
        )

        return connection
    except OSError:
        return None
    except sqlite3.Error:
        return None

def _lookup_cached_embedding(
    connection: sqlite3.Connection,
    embedding_model: str,
    input_hash: str
) -> NDArray[float32] | None:
    row = connection.execute(
        """
        SELECT embedding
        FROM embedding_cache
        WHERE embedding_model = ?
          AND input_hash = ?
          AND schema_version = ?
        """,
        (embedding_model, input_hash, cache_schema_version)
    ).fetchone()

    if row is None:
        return None

    try:
        return numpy.frombuffer(row[0], dtype = float32).copy()
    except ValueError:
        return None

def _store_cached_embeddings(
    connection: sqlite3.Connection,
    embedding_model: str,
    rows: list[tuple[EmbedInput, NDArray[float32]]]
) -> None:
    if not rows:
        return

    connection.executemany(
        """
        INSERT OR REPLACE INTO embedding_cache (
            embedding_model,
            input_hash,
            file_hash,
            path,
            chunk_index,
            schema_version,
            embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                embedding_model,
                input.input_hash,
                input.file_hash,
                input.entry,
                input.chunk_index,
                cache_schema_version,
                sqlite3.Binary(embedding.tobytes())
            )
            for input, embedding in rows
        ]
    )

    connection.commit()

def _subdirectory(repo: Repo, directory: str) -> PurePath:
    target = PurePath(directory)
    repo_path = PurePath(repo.path)

    for candidate in [repo_path, repo_path.parent]:
        try:
            return target.relative_to(candidate)
        except ValueError:
            pass

    raise ValueError(
        f"{directory!r} is not below git repository path {repo.path!r}"
    )

def tracked_paths(directory: str) -> list[str]:
    repo = Repo.discover(directory)
    subdirectory = _subdirectory(repo, directory)
    paths = [ ]

    for bytestring in repo.open_index().paths():
        try:
            path = bytestring.decode("utf-8")
        except UnicodeDecodeError:
            continue

        try:
            relative_path = PurePath(path).relative_to(subdirectory)
        except ValueError:
            continue

        absolute_path = os.path.join(directory, str(relative_path))

        if os.path.isfile(absolute_path):
            paths.append(str(relative_path))

    return paths

def confirm_token_spend(file_count: int) -> bool:
    if file_count < token_confirmation_threshold:
        return True

    prompt = (
        f"This run will embed {file_count} tracked files and may burn tokens. "
        "Continue? [y/N]: "
    )
    reply = input(prompt).strip().lower()

    return reply in [ "y", "yes" ]

async def embed(facets: Facets, directory: str, paths: Iterable[str]) -> Cluster:

    async def read(path: str) -> list[EmbedInput]:
        try:
            absolute_path = os.path.join(directory, path)

            async with facets.semaphore, aiofiles.open(absolute_path, "rb") as handle:
                prefix = f"{path}:\n\n"

                bytestring = await handle.read()

                file_hash = hashlib.sha256(bytestring).hexdigest()

                text = bytestring.decode("utf-8")

                prefix_tokens = facets.embedding_encoding.encode(prefix)
                text_tokens   = facets.embedding_encoding.encode(text)

                max_tokens_per_chunk = max_tokens_per_embed - len(prefix_tokens)

                inputs = []

                # TODO: This currently only takes the first chunk because
                # GPT has trouble labeling chunks in order when multiple
                # chunks have the same file name.  Remove the `[:1]` when
                # this is fixed.
                for index, chunk in enumerate(list(batched(text_tokens, max_tokens_per_chunk))[:1]):
                    content = facets.embedding_encoding.decode(prefix_tokens + list(chunk))
                    input_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    inputs.append(
                        EmbedInput(
                            entry = path,
                            content = content,
                            file_hash = file_hash,
                            input_hash = input_hash,
                            chunk_index = index
                        )
                    )

                return inputs

        except UnicodeDecodeError:
            # Ignore files that aren't UTF-8
            return [ ]

        except IsADirectoryError:
            # This can happen when a "file" listed by the repository is:
            #
            # - a submodule
            # - a symlink to a directory
            #
            # TODO: The submodule case can and should be fixed and properly
            # handled
            return [ ]

        except FileNotFoundError:
            # Ignore files that have been removed from the working tree.
            return [ ]

        except PermissionError:
            # Ignore files that cannot be read.
            return [ ]

    tasks = tqdm_asyncio.gather(
        *(read(path) for path in paths),
        desc = "Reading files",
        unit = "file",
        leave = False
    )

    inputs = list(chain.from_iterable(await tasks))

    if not inputs:
        return Cluster([])

    max_embeds = math.floor(max_tokens_per_batch_embed / max_tokens_per_embed)

    @retry(retry = tenacity.retry_if_exception_type(RateLimitError), wait = tenacity.wait_fixed(1), stop = tenacity.stop_after_attempt(3))
    async def embed_batch(input: tuple[str, ...]) -> list[NDArray[float32]]:
        response = await facets.embedding_client.embeddings.create(
            model = facets.embedding_model,
            input = input
        )

        outputs = [
            numpy.asarray(datum.embedding, float32) for datum in response.data
        ]

        return outputs

    connection = _open_cache_connection()
    embeddings: list[NDArray[float32] | None] = [None] * len(inputs)
    missing_indices: list[int] = []

    try:
        if connection is None:
            missing_indices = list(range(len(inputs)))
        else:
            for index, input in enumerate(inputs):
                cached_embedding = _lookup_cached_embedding(
                    connection,
                    facets.embedding_model,
                    input.input_hash
                )

                if cached_embedding is None:
                    missing_indices.append(index)
                else:
                    embeddings[index] = cached_embedding

        if missing_indices:
            contents = tuple(inputs[index].content for index in missing_indices)

            tasks = tqdm_asyncio.gather(
                *(embed_batch(input) for input in batched(contents, max_embeds)),
                desc = "Embedding contents",
                unit = "batch",
                leave = False
            )

            fresh_embeddings = list(chain.from_iterable(await tasks))

            for index, embedding in zip(missing_indices, fresh_embeddings):
                embeddings[index] = embedding

            if connection is not None:
                _store_cached_embeddings(
                    connection,
                    facets.embedding_model,
                    [
                        (inputs[index], embedding)
                        for index, embedding in zip(missing_indices, fresh_embeddings)
                    ]
                )
    finally:
        if connection is not None:
            connection.close()

    assert all(embedding is not None for embedding in embeddings)

    embeds = [
        Embed(input.entry, input.content, embedding)
        for input, embedding in zip(inputs, embeddings)
        if embedding is not None
    ]

    return Cluster(embeds)

# The clustering algorithm can go as low as 1 here, but we set it higher for
# two reasons:
#
# - it's easier for users to navigate when there is more branching at the
#   leaves
# - this also avoids straining the tree visualizer, which doesn't like a really
#   deeply nested tree structure.
max_leaves = 20

def cluster(input: Cluster) -> list[Cluster]:
    N = len(input.embeds)

    if N <= max_leaves:
        return [input]

    entries, contents, embeddings = zip(*(
        (embed.entry, embed.content, embed.embedding)
        for embed in input.embeds
    ))

    # The following code computes an affinity matrix using a radial basis
    # function with an adaptive σ.  See:
    #
    #     L. Zelnik-Manor, P. Perona (2004), "Self-Tuning Spectral Clustering"

    normalized = sklearn.preprocessing.normalize(embeddings)

    # The original paper suggests setting K (`n_neighbors`) to 7.  Here we do
    # something a little fancier and try to find a low value of `n_neighbors`
    # that produces one connected component.  This usually ends up being around
    # 7 anyway.
    #
    # The reason we want to avoid multiple connected components is because if
    # we have more than one connected component then those connected components
    # will dominate the clusters suggested by spectral clustering.  We don't
    # want that because we don't want spectral clustering to degenerate to the
    # same result as K nearest neighbors.  We want the K nearest neighbors
    # algorithm to weakly inform the spectral clustering algorithm without
    # dominating the result.
    def get_nearest_neighbors(n_neighbors: int) -> tuple[int, int, NearestNeighbors]:
        nearest_neighbors = NearestNeighbors(
            n_neighbors = n_neighbors,
            metric = "cosine",
            n_jobs = -1
        ).fit(normalized)

        graph = nearest_neighbors.kneighbors_graph(
            mode = "connectivity"
        )

        n_components, _ = scipy.sparse.csgraph.connected_components(
            graph,
            directed = False
        )

        return n_components, n_neighbors, nearest_neighbors

    # We don't attempt to find the absolute lowest value of K (`n_neighbors`).
    # Instead we just sample a few values and pick a "small enough" one.
    candidate_neighbor_counts = list(itertools.takewhile(
        lambda x: x < N,
        (round(math.exp(n)) for n in itertools.count())
    )) + [ math.floor(N / 2) ]

    results = [
        get_nearest_neighbors(n_neighbors)
        for n_neighbors in candidate_neighbor_counts
    ]

    # Find the first sample value of K (`n_neighbors`) that produces one
    # connected component.  There's guaranteed to be at least one since the
    # very last value we sample (⌊N/2⌋) always produces one connected
    # component.
    n_neighbors, nearest_neighbors = [
        (n_neighbors, nearest_neighbors)
        for n_components, n_neighbors, nearest_neighbors in results
        if n_components == 1
    ][0]

    distances, indices = nearest_neighbors.kneighbors()

    # sigmas[i] = the distance of semantic embedding #i to its Kth nearest
    # neighbor
    sigmas = distances[:, -1]

    rows    = numpy.repeat(numpy.arange(N), n_neighbors)
    columns = indices.reshape(-1)

    d = distances.reshape(-1)

    sigma_i = numpy.repeat(sigmas, n_neighbors)
    sigma_j = sigmas[columns]

    denominator = numpy.maximum(sigma_i * sigma_j, 1e-12)
    data = numpy.exp(-(d * d) / denominator).astype(numpy.float32)

    # Affinity: A_ij = exp(-d(x_i, x_j)^2 / (σ_i σ_j))
    affinity = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (N, N)).tocsr()

    affinity = (affinity + affinity.T) * 0.5
    affinity.setdiag(1.0)
    affinity.eliminate_zeros()

    # The following code is basically `sklearn.manifold.spectral_embeddings`,
    # but exploded out so that we can get access to the eigenvalues, which are
    # normally not exposed by the function.  We'll need those eigenvalues
    # later.
    random_state = sklearn.utils.check_random_state(0)

    laplacian, dd = scipy.sparse.csgraph.laplacian(
        affinity,
        normed = True,
        return_diag = True
    )

    # laplacian = set_diag(laplacian, 1, True)
    laplacian = laplacian.tocoo()
    laplacian.data[laplacian.row == laplacian.col] = 1
    laplacian = laplacian.tocsr()

    laplacian *= -1
    v0 = random_state.uniform(-1, 1, N)

    if max_clusters + 1 < N:
        k = max_clusters + 1

        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            laplacian,
            k = k,
            sigma = 1.0,
            which = 'LM',
            tol = 0.0,
            v0 = v0
        )
    else:
        k = N

        eigenvalues, eigenvectors = scipy.linalg.eigh(
            laplacian.toarray(),
            check_finite = False
        )

    indices = numpy.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]

    eigenvectors = eigenvectors[:, indices]

    wide_spectral_embeddings = eigenvectors.T / dd
    wide_spectral_embeddings = sklearn.utils.extmath._deterministic_vector_sign_flip(wide_spectral_embeddings)
    wide_spectral_embeddings = wide_spectral_embeddings[1:k].T
    eigenvalues = eigenvalues * -1

    # Find the optimal cluster count by looking for the largest eigengap
    #
    # The reason the suggested cluster count is not just:
    #
    #     numpy.argmax(numpy.diff(eigenvalues)) + 1
    #
    # … is because we want at least two clusters
    n_clusters = numpy.argmax(numpy.diff(eigenvalues[1:])) + 2

    spectral_embeddings = wide_spectral_embeddings[:, :n_clusters]

    spectral_embeddings = sklearn.preprocessing.normalize(spectral_embeddings)

    labels = sklearn.cluster.KMeans(
        n_clusters = n_clusters,
        random_state = 0,
        n_init = "auto"
    ).fit_predict(spectral_embeddings)

    groups = collections.OrderedDict()

    for (label, entry, content, embedding) in zip(labels, entries, contents, embeddings):
        groups.setdefault(label, []).append(Embed(entry, content, embedding))

    return [ Cluster(embeds) for embeds in groups.values() ]

@dataclass(frozen = True)
class Tree:
    label: str
    files: list[str]
    children: list["Tree"]

def to_pattern(files: list[str]) -> str:
    prefix = os.path.commonprefix(files)
    suffix = os.path.commonprefix([ file[len(prefix):][::-1] for file in files ])[::-1]

    if suffix:
        if any([ file[len(prefix):-len(suffix)] for file in files ]):
            star = "*"
        else:
            star = ""
    else:
        if any([ file[len(prefix):] for file in files ]):
            star = "*"
        else:
            star = ""

    if prefix:
        if suffix:
            return f"{prefix}{star}{suffix}: "
        else:
            return f"{prefix}{star}: "
    else:
        if suffix:
            return f"{star}{suffix}: "
        else:
            return ""

class Label(BaseModel):
    overarchingTheme: str
    distinguishingFeature: str
    label: str
    
class Labels(BaseModel):
    labels: list[Label]

def to_files(trees: list[Tree]) -> list[str]:
    return [ file for tree in trees for file in tree.files ]

async def label_nodes(facets: Facets, c: Cluster, depth: int) -> list[Tree]:
    children = cluster(c)

    @retry(retry = tenacity.retry_if_exception_type(RateLimitError), wait = tenacity.wait_fixed(1), stop = tenacity.stop_after_attempt(3))
    async def label_inputs(input: str, expected_length: int) -> list[str]:
        h = hashlib.sha256(facets.completion_model.encode("utf-8"))

        h.update(input.encode("utf-8"))

        cache_file = os.path.join(facets.cache_directory, h.hexdigest())

        @retry(retry = tenacity.retry_if_exception_type(AssertionError), stop = tenacity.stop_after_attempt(3))
        async def compute() -> list[str]:
            response = await facets.completion_client.chat.completions.parse(
                model = facets.completion_model,
                messages = [
                    { "role": "user", "content": input },
                ],
                response_format = Labels
            )

            text = response.choices[0].message.content

            assert text is not None

            parsed = Labels.model_validate_json(text)

            labels = [ label.label for label in parsed.labels ]

            assert len(labels) == expected_length

            return labels

        if os.path.exists(cache_file):
            async with facets.semaphore, aiofiles.open(cache_file, "r") as handle:
                try:
                    labels = json.loads(await handle.read())
                except JSONDecodeError:
                    labels = await compute()
        else:
            labels = await compute()

        async with facets.semaphore, aiofiles.open(cache_file, "w") as handle:
            await handle.write(json.dumps(labels))

        return labels

    if len(children) == 1:
        def render_embed(embed: Embed) -> str:
            return f"# File: {embed.entry}\n\n{embed.content}"

        rendered_embeds = "\n\n".join([ render_embed(embed) for embed in c.embeds ])

        input = f"Label each file in 3 to 7 words.  Don't include file path/names in descriptions.\n\n{rendered_embeds}"

        labels = await label_inputs(input, len(c.embeds))

        return [
            Tree(f"{embed.entry}: {label}", [ embed.entry ], [])
            for label, embed in zip(labels, c.embeds)
        ]

    else:
        if depth == 0:
            treess = await tqdm_asyncio.gather(
                *(label_nodes(facets, child, depth + 1) for child in children),
                desc = "Labeling clusters",
                unit = "cluster",
                leave = False
            )
        else:
            treess = await asyncio.gather(
                *(label_nodes(facets, child, depth + 1) for child in children),
            )

        def render_cluster(trees: list[Tree]) -> str:
            rendered_trees = "\n".join([ tree.label for tree in trees ])

            return f"# Cluster\n\n{rendered_trees}"

        rendered_clusters = "\n\n".join([ render_cluster(trees) for trees in treess ])

        input = f"Label each cluster in 2 words.  Don't include file path/names in labels.\n\n{rendered_clusters}"

        labels = await label_inputs(input, len(treess))

        return [
            Tree(f"{to_pattern(to_files(trees))}{label}", to_files(trees), trees)
            for label, trees in zip(labels, treess)
        ]

async def tree(facets: Facets, label: str, c: Cluster) -> Tree:
    children = await label_nodes(facets, c, 0)

    return Tree(label, to_files(children), children)

def filtered_trees(filter_text: str, t: Tree) -> list[Tree]:
    new_children = [
        filtered_tree
        for child in t.children
        for filtered_tree in filtered_trees(filter_text, child)
    ]

    if new_children or filter_text in t.label.lower():
        return [ Tree(t.label, t.files, new_children) ]
    else:
        return [ ]

class UI(textual.app.App):
    BINDINGS = [
        ("slash", "focus_search", "Search"),
        ("escape", "exit_search", "Clear"),
    ]

    def __init__(self, tree_):
        super().__init__()

        self.search = textual.widgets.Input(
            placeholder="Type / to search"
        )

        self.tree_ = tree_

    def _build_tree(self, filter_text: str = ""):
        self.treeview.clear()

        trees = filtered_trees(filter_text, self.tree_)

        if trees:
            filtered = trees[0]
        else:
            filtered = Tree(self.tree_.label, [], [])

        self.treeview.root.set_label(f"{filtered.label} ({len(filtered.files)})")

        def loop(node, children):
            for child in children:
                if len(child.files) <= 1:
                    n = node.add(child.label, allow_expand = False)
                else:
                    n = node.add(
                        f"{child.label} ({len(child.files)})",
                        allow_expand = True
                    )

                    loop(n, child.children)

        _ = loop(self.treeview.root, filtered.children)

        if filter_text:
            self.treeview.root.expand_all()
        else:
            self.treeview.root.expand()

    async def on_mount(self):
        self.treeview = textual.widgets.Tree(f"{self.tree_.label} ({len(self.tree_.files)})")

        self._build_tree()

        self.mount(self.search)
        self.mount(self.treeview)

        self.treeview.focus()

    def action_focus_search(self):
        self.search.focus()
        self.search.placeholder = "Type ESC to navigate"

    def action_exit_search(self):
        self.search.placeholder = "Type / to search"
        self.treeview.focus()

    def on_input_changed(self, event):
        self._build_tree(event.value.strip().lower())

def main():
    parser = argparse.ArgumentParser(
        prog = "semantic-navigator",
        description = "Cluster documents by semantic facets",
    )

    parser.add_argument("repository", nargs = "?", default = ".")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--completion-base-url")
    parser.add_argument("--embedding-model", default = "text-embedding-3-large")
    parser.add_argument("--completion-model", default = "gpt-5-mini")
    parser.add_argument("--completion-encoding")
    parser.add_argument("--embedding-encoding")
    arguments = parser.parse_args()

    directory = os.path.abspath(arguments.repository)

    try:
        paths = tracked_paths(directory)
    except NotGitRepository:
        parser.error(f"{directory!r} is not a git repository directory")
    except ValueError as error:
        parser.error(str(error))

    if not confirm_token_spend(len(paths)):
        print("Canceled.")
        return

    facets = initialize(
        arguments.embedding_model,
        arguments.completion_model,
        embedding_base_url = arguments.embedding_base_url,
        completion_base_url = arguments.completion_base_url,
        embedding_encoding_name = arguments.embedding_encoding,
        completion_encoding_name = arguments.completion_encoding,
    )

    async def async_tasks():
        initial_cluster = await embed(facets, directory, paths)

        tree_ = await tree(facets, directory, initial_cluster)

        return tree_

    tree_ = asyncio.run(async_tasks())

    UI(tree_).run()

if __name__ == "__main__":
    main()
