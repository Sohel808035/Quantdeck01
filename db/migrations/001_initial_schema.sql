-- QuantSphereX V2 Database Architecture
-- PostgreSQL + TimescaleDB Migration Script
-- Version: 001_initial_schema.sql

-- Enable TimescaleDB extension if available
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- -----------------------------------------------------------------------------
-- 1. MARKET DATA HYPERTABLE (OHLCV Time-Series)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_data (
    timestamp       TIMESTAMPTZ NOT NULL,
    ticker          VARCHAR(30)  NOT NULL,
    open_price      NUMERIC(14, 4),
    high_price      NUMERIC(14, 4),
    low_price       NUMERIC(14, 4),
    close_price     NUMERIC(14, 4) NOT NULL,
    volume          BIGINT,
    roe             NUMERIC(8, 4),
    roa             NUMERIC(8, 4),
    earnings_growth NUMERIC(8, 4),
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_market_data PRIMARY KEY (timestamp, ticker)
);

-- Convert market_data into TimescaleDB Hypertable partitioned by time
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Create compound index for fast symbol and date range lookup
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_time ON market_data (ticker, timestamp DESC);


-- -----------------------------------------------------------------------------
-- 2. MACRO DATA HYPERTABLE (NIFTY 50, India VIX, Interest Rates)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS macro_data (
    timestamp   TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(30)  NOT NULL,
    open_price  NUMERIC(14, 4),
    high_price  NUMERIC(14, 4),
    low_price   NUMERIC(14, 4),
    close_price NUMERIC(14, 4) NOT NULL,
    volume      BIGINT,
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_macro_data PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('macro_data', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_macro_data_symbol_time ON macro_data (symbol, timestamp DESC);


-- -----------------------------------------------------------------------------
-- 3. POINT-IN-TIME (PIT) UNIVERSE MEMBERSHIP TABLE
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS universe_membership (
    timestamp       TIMESTAMPTZ NOT NULL,
    ticker          VARCHAR(30)  NOT NULL,
    universe_name   VARCHAR(50)  DEFAULT 'NIFTY200',
    is_in_universe  BOOLEAN      NOT NULL DEFAULT TRUE,
    sector          VARCHAR(100) DEFAULT 'Other / Midcap',
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_universe_membership PRIMARY KEY (timestamp, ticker, universe_name)
);

CREATE INDEX IF NOT EXISTS idx_universe_membership_lookup ON universe_membership (universe_name, timestamp DESC, ticker);


-- -----------------------------------------------------------------------------
-- 4. DATASET METADATA & VERSIONING MANIFEST TABLE
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dataset_metadata (
    version_id      VARCHAR(50)  NOT NULL PRIMARY KEY,
    dataset_name    VARCHAR(100) NOT NULL,
    hash_key        VARCHAR(32)  NOT NULL,
    row_count       BIGINT       NOT NULL,
    start_date      DATE         NOT NULL,
    end_date        DATE         NOT NULL,
    file_path       TEXT         NOT NULL,
    schema_json     JSONB,
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dataset_metadata_name_version ON dataset_metadata (dataset_name, version_id);


-- -----------------------------------------------------------------------------
-- 5. MODEL VERSIONS & REGISTRY METADATA TABLE
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    model_id        VARCHAR(64)  NOT NULL PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,
    algorithm       VARCHAR(50)  NOT NULL,
    version         VARCHAR(20)  NOT NULL,
    train_ic        NUMERIC(8, 4),
    val_ic          NUMERIC(8, 4),
    hyperparams     JSONB,
    artifact_path   TEXT,
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
