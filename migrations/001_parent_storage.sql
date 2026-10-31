CREATE TABLE document_ingestions (
    document_id text NOT NULL,
    document_hash text NOT NULL,
    chunking_config_hash text NOT NULL,
    status text NOT NULL CHECK (status IN ('processing', 'completed')),
    expected_parents integer NOT NULL CHECK (expected_parents > 0),
    expected_children integer NOT NULL CHECK (expected_children > 0),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, document_hash, chunking_config_hash)
);

CREATE TABLE parent_chunks (
    id uuid PRIMARY KEY,
    document_id text NOT NULL,
    document_hash text NOT NULL,
    chunking_config_hash text NOT NULL,
    document_code text NOT NULL,
    title text NOT NULL,
    file_name text NOT NULL,
    version text NOT NULL,
    source_path text NOT NULL,
    source_url text NOT NULL,
    section text NOT NULL,
    heading_path text[] NOT NULL,
    text text NOT NULL CHECK (length(text) > 0),
    parent_index integer NOT NULL CHECK (parent_index >= 0),
    source_start integer NOT NULL CHECK (source_start >= 0),
    source_end integer NOT NULL CHECK (source_end > source_start),
    parent_hash text NOT NULL,
    markdown_prefix text NOT NULL,
    markdown_suffix text NOT NULL,
    FOREIGN KEY (document_id, document_hash, chunking_config_hash)
        REFERENCES document_ingestions (document_id, document_hash, chunking_config_hash)
);

CREATE INDEX parent_chunks_revision_idx
    ON parent_chunks (document_id, document_hash, chunking_config_hash);
