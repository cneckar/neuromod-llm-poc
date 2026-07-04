-- Chat archive for the neuromod demo (Cloudflare D1).
-- Apply:  npx wrangler d1 execute neuromod_chats --file=migrations/0001_create_chats.sql
-- (add --remote for the deployed DB; omit for the local dev DB).

CREATE TABLE IF NOT EXISTS chats (
  id          TEXT PRIMARY KEY,   -- crypto.randomUUID()
  created     TEXT NOT NULL,      -- ISO-8601 timestamp
  tier        TEXT,               -- 'pro' | 'default' (server-side only; never sent to the browser)
  task        TEXT,               -- 'chat' | 'image'
  pack_name   TEXT,               -- applied predefined pack, if any
  custom_pack TEXT,               -- JSON of a chemistry-lab custom pack, if any
  intensity   REAL,
  had_image   INTEGER DEFAULT 0,  -- 1 if the exchange produced an image (bytes are NOT stored)
  messages    TEXT,               -- JSON array of the user-visible conversation turns
  assistant   TEXT,               -- assembled assistant reply
  reasoning   TEXT,               -- gpt-oss analysis channel, if any
  error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_chats_created ON chats (created DESC);
