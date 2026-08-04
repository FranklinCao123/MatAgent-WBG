-- Atomic document + chunk ingestion for the RAG evidence store.

create or replace function public.ingest_rag_document(
    document_title text,
    document_source_type text,
    document_source_url text,
    document_doi text,
    document_publisher text,
    document_publication_year integer,
    document_abstract text,
    document_metadata jsonb,
    chunks jsonb
)
returns uuid
language plpgsql
volatile
security invoker
set search_path = ''
as $$
declare
    stored_document_id uuid;
    normalized_doi text := nullif(btrim(document_doi), '');
begin
    if jsonb_typeof(chunks) is distinct from 'array'
       or jsonb_array_length(chunks) = 0 then
        raise exception 'chunks must be a non-empty JSON array'
            using errcode = '22023';
    end if;
    if jsonb_array_length(chunks) > 500 then
        raise exception 'a document cannot contain more than 500 chunks'
            using errcode = '22023';
    end if;

    if normalized_doi is not null then
        select document.id
        into stored_document_id
        from public.rag_documents as document
        where lower(document.doi) = lower(normalized_doi)
        for update;
    end if;

    if stored_document_id is null then
        insert into public.rag_documents (
            title,
            source_type,
            source_url,
            doi,
            publisher,
            publication_year,
            abstract,
            metadata
        )
        values (
            document_title,
            document_source_type,
            document_source_url,
            normalized_doi,
            document_publisher,
            document_publication_year,
            document_abstract,
            coalesce(document_metadata, '{}'::jsonb)
        )
        returning id into stored_document_id;
    else
        update public.rag_documents
        set
            title = document_title,
            source_type = document_source_type,
            source_url = document_source_url,
            publisher = document_publisher,
            publication_year = document_publication_year,
            abstract = document_abstract,
            metadata = coalesce(document_metadata, '{}'::jsonb)
        where id = stored_document_id;

        delete from public.rag_document_chunks
        where document_id = stored_document_id;
    end if;

    insert into public.rag_document_chunks (
        document_id,
        chunk_index,
        content,
        token_count,
        embedding,
        material_names,
        metadata
    )
    select
        stored_document_id,
        (item.element->>'chunk_index')::integer,
        item.element->>'content',
        (item.element->>'token_count')::integer,
        (item.element->>'embedding')::extensions.vector(1024),
        coalesce(
            array(
                select jsonb_array_elements_text(
                    item.element->'material_names'
                )
            ),
            '{}'::text[]
        ),
        coalesce(item.element->'metadata', '{}'::jsonb)
    from jsonb_array_elements(chunks) as item(element);

    return stored_document_id;
end;
$$;

revoke all on function public.ingest_rag_document(
    text, text, text, text, text, integer, text, jsonb, jsonb
) from public, anon, authenticated;
grant execute on function public.ingest_rag_document(
    text, text, text, text, text, integer, text, jsonb, jsonb
) to service_role;

notify pgrst, 'reload schema';
