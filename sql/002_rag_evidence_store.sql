-- Production-shaped evidence storage for multilingual scientific RAG.
-- The 1024 dimensions match the planned BAAI/bge-m3 embedding contract.

create table if not exists public.rag_documents (
    id uuid primary key default gen_random_uuid(),
    title text not null check (length(btrim(title)) > 0),
    source_type text not null default 'paper'
        check (source_type in ('paper', 'web', 'dataset', 'manual')),
    source_url text,
    doi text,
    publisher text,
    publication_year integer check (publication_year between 1800 and 2100),
    abstract text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create unique index if not exists rag_documents_doi_unique
    on public.rag_documents (lower(doi))
    where doi is not null;

create table if not exists public.rag_document_chunks (
    id bigint generated always as identity primary key,
    document_id uuid not null
        references public.rag_documents(id) on delete cascade,
    chunk_index integer not null check (chunk_index >= 0),
    content text not null check (length(btrim(content)) > 0),
    token_count integer check (token_count > 0),
    embedding extensions.vector(1024) not null,
    material_names text[] not null default '{}',
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (document_id, chunk_index)
);

create index if not exists rag_document_chunks_document_id_idx
    on public.rag_document_chunks (document_id);

create index if not exists rag_document_chunks_material_names_idx
    on public.rag_document_chunks using gin (material_names);

create index if not exists rag_document_chunks_embedding_hnsw_idx
    on public.rag_document_chunks
    using hnsw (embedding extensions.vector_cosine_ops);

alter table public.rag_documents enable row level security;
alter table public.rag_document_chunks enable row level security;

revoke all on table public.rag_documents from anon, authenticated;
revoke all on table public.rag_document_chunks from anon, authenticated;
grant select, insert, update, delete on table public.rag_documents to service_role;
grant select, insert, update, delete on table public.rag_document_chunks to service_role;
grant usage, select on sequence public.rag_document_chunks_id_seq to service_role;

create or replace function public.match_rag_chunks(
    query_embedding extensions.vector(1024),
    match_count integer default 5,
    match_threshold double precision default 0.0,
    material_filter text default null
)
returns table (
    chunk_id bigint,
    document_id uuid,
    title text,
    content text,
    source_url text,
    doi text,
    publication_year integer,
    material_names text[],
    metadata jsonb,
    similarity double precision
)
language sql
stable
security invoker
set search_path = ''
as $$
    select
        chunk.id,
        document.id,
        document.title,
        chunk.content,
        document.source_url,
        document.doi,
        document.publication_year,
        chunk.material_names,
        chunk.metadata,
        1 - (
            chunk.embedding OPERATOR(extensions.<=>) query_embedding
        ) as similarity
    from public.rag_document_chunks as chunk
    join public.rag_documents as document on document.id = chunk.document_id
    where
        (
            material_filter is null
            or chunk.material_names @> array[material_filter]
        )
        and 1 - (
            chunk.embedding OPERATOR(extensions.<=>) query_embedding
        ) >= match_threshold
    order by chunk.embedding OPERATOR(extensions.<=>) query_embedding
    limit least(greatest(match_count, 1), 50);
$$;

revoke all on function public.match_rag_chunks(
    extensions.vector, integer, double precision, text
) from public, anon, authenticated;
grant execute on function public.match_rag_chunks(
    extensions.vector, integer, double precision, text
) to service_role;

notify pgrst, 'reload schema';
