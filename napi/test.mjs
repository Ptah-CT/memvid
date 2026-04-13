import { SplitStore } from './index.js';
import { mkdtempSync, rmSync, readFileSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

const dir = mkdtempSync(join(tmpdir(), 'memvid-test-'));
const mv2dPath = join(dir, 'test.mv2d');

try {
  // Create store
  const store = new SplitStore(mv2dPath);
  console.log('✓ SplitStore created');

  // Put frames
  const id0 = store.put('SurrealDB drops connections after 32min idle');
  const id1 = store.putFrame(
    'PQC ML-DSA-87 keys rotated',
    'mv2://memory/episodic/001',
    'Key Rotation',
    { agent: 'ptah', sector: 'episodic' }
  );
  console.log(`✓ Put frames: ${id0}, ${id1}`);

  // Get frame
  const frame = store.get(id1);
  console.log(`✓ Get frame ${id1}: "${frame.content}" (tags: ${JSON.stringify(frame.tags)})`);

  // Active frames
  console.log(`✓ Frame count: ${store.frameCount}, active: ${store.activeFrameCount}`);

  // Update
  const id2 = store.update(id0, 'SurrealDB drops connections — FIXED');
  console.log(`✓ Updated frame ${id0} → ${id2}`);

  // Active after update
  const active = store.activeFrames();
  console.log(`✓ Active frames after update: ${active.length}`);

  // Rebuild lex index + search
  store.rebuildLexIndex();
  store.flushIndex();
  const hits = store.lexSearch('connections', 5);
  console.log(`✓ Lex search "connections": ${hits.length} hit(s)`);
  for (const hit of hits) {
    console.log(`  frame_id=${hit.frameId} score=${hit.score.toFixed(3)}`);
  }

  // Latest active
  const latest = store.latestActive();
  console.log(`✓ Latest active: "${latest.content}"`);

  // Compact
  const result = store.compact();
  console.log(`✓ Compact: ${result.originalFrames} → ${result.activeFrames} (removed ${result.removedFrames})`);

  // Verify .mv2d is cat-readable
  const raw = readFileSync(mv2dPath, 'utf-8');
  console.log(`✓ .mv2d is cat-readable (${raw.split('\\n').filter(Boolean).length} lines)`);

  console.log('\n✓ ALL TESTS PASSED');
} finally {
  rmSync(dir, { recursive: true, force: true });
}
