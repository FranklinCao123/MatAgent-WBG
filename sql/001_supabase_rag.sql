-- Minimal server-only health RPC for the MatAgent RAG database.
create extension if not exists vector;

create or replace function public.matagent_database_health()
returns table (
    database_name text,
    postgres_version text,
    vector_version text
)
language sql
stable
security invoker
set search_path = ''
as $$
    select
        current_database()::text,
        current_setting('server_version')::text,
        (
            select extension.extversion::text
            from pg_catalog.pg_extension as extension
            where extension.extname = 'vector'
        );
$$;

revoke all on function public.matagent_database_health() from public;
revoke all on function public.matagent_database_health() from anon;
revoke all on function public.matagent_database_health() from authenticated;
grant execute on function public.matagent_database_health() to service_role;
