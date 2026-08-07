/**
 * Preview every glob state without a microphone, an LLM, or the daemon.
 * Stands up a fake daemon, drives the UI through each state, screenshots it.
 * Also the fastest way to check a shader change didn't break anything.
 *
 *   npm run build
 *   npm i --no-save playwright ws
 *   node state-test.mjs          # writes glob-<state>.png to OUT_DIR (default /tmp)
 */
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { WebSocketServer } from 'ws';

const ROOT = join(import.meta.dirname, 'dist');
const OUT = process.env.OUT_DIR ?? '/tmp';
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };

const http = createServer(async (req, res) => {
  const path = join(ROOT, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
  try {
    const body = await readFile(path);
    res.writeHead(200, { 'Content-Type': TYPES[extname(path)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => http.listen(4173, r));

const wss = new WebSocketServer({ port: 8765, host: '127.0.0.1' });
let live = null;
wss.on('connection', (ws) => { live = ws; });
const send = (type, payload) => live?.send(JSON.stringify({ type, payload }));

const browser = await chromium.launch({
  // CHROME_PATH is only needed in sandboxes without a bundled browser; on your
  // Mac playwright finds its own.
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 620, height: 620 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (/shader|glsl|invalid/i.test(m.text())) errors.push(m.text()); });

await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle' });
await page.waitForTimeout(800);

const results = [];
for (const [state, level] of [['idle', 0], ['listening', 0.55], ['thinking', 0], ['speaking', 0.6], ['working', 0]]) {
  send('state', { state });
  for (let i = 0; i < 14; i++) { send('level', { rms: level }); await page.waitForTimeout(90); }
  await page.waitForTimeout(600);
  await page.screenshot({ path: join(OUT, `glob-${state}.png`) });
  results.push(state);
}

console.log('page errors:', errors.length ? errors : 'none');
for (const state of results) console.log(`  wrote glob-${state}.png`);

await browser.close();
http.close();
wss.close();
