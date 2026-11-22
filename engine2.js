// 정보통신설비 성능점검 — 현황사진 자동 배치 엔진 v2 (모든 처리는 이 PC 브라우저 안에서만)
import * as ort from './ort.wasm.bundle.min.mjs?v=20261122a';

const NS = {
  w: 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
  wp: 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
  a: 'http://schemas.openxmlformats.org/drawingml/2006/main',
  pic: 'http://schemas.openxmlformats.org/drawingml/2006/picture',
  r: 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
  rel: 'http://schemas.openxmlformats.org/package/2006/relationships',
  ct: 'http://schemas.openxmlformats.org/package/2006/content-types',
};
export const DPI = 200;          // 워드에 넣는 사진 해상도 (인쇄 기준 충분)
export const JPEG_Q = 0.8;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
let session = null;

// ------------------------------------------------------------ 모델
async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + ' 불러오기 실패 (' + res.status + ')');
  const total = +res.headers.get('content-length') || 0;
  if (!res.body || !res.body.getReader) { const b = await res.arrayBuffer(); onBytes(b.byteLength, total); return b; }
  const reader = res.body.getReader(); const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onBytes(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}
async function fetchModel(onBytes) {
  const got = [0, 0], tot = [0, 0];
  const bufs = await Promise.all([0, 1].map(i => fetchWithProgress(`./model_${i}.json`, (g, t) => {
    got[i] = g; tot[i] = t || tot[i] || 10679962; onBytes(got[0] + got[1], tot[0] + tot[1] || 21359924);
  })));
  const bytes = bufs.map(b => { const j = JSON.parse(new TextDecoder().decode(b)); const s = atob(j.b64); const u = new Uint8Array(s.length); for (let k = 0; k < s.length; k++) u[k] = s.charCodeAt(k); return u; });
  const out = new Uint8Array(bytes[0].length + bytes[1].length); out.set(bytes[0], 0); out.set(bytes[1], bytes[0].length);
  return out.buffer;
}
export async function loadModel(onProgress) {
  if (session) return;
  const parts = { wasm: [0, 14848852], model: [0, 21359924] };
  const tick = () => onProgress && onProgress(parts.wasm[0] + parts.model[0], parts.wasm[1] + parts.model[1]);
  const [wasm, model] = await Promise.all([
    fetchWithProgress('./ort-wasm-simd-threaded.wasm', (g, t) => { parts.wasm = [g, t || parts.wasm[1]]; tick(); }),
    fetchModel((g, t) => { parts.model = [g, t]; tick(); }),
  ]);
  ort.env.wasm.wasmBinary = wasm;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: ['wasm'] });
}
export async function loadExtraRefs() {
  try {
    const j = await (await fetch('./refs_extra.json?v=20261122a')).json();
    return j.refs.map(r => ({ label: normFac(r.label), emb: b64ToF32(r.emb), src: 'extra' }));
  } catch (e) { return []; }
}
export function b64ToF32(b64) {
  const s = atob(b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
}
function l2(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; for (let i = 0; i < v.length; i++) v[i] /= s; return v; }
export function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

// 짧은 변 256 캔버스 → 가운데 224 → 특징값
export async function embedCanvas(cv) {
  const g = cv.getContext('2d');
  const ox = Math.floor((cv.width - 224) / 2), oy = Math.floor((cv.height - 224) / 2);
  const px = g.getImageData(Math.max(0, ox), Math.max(0, oy), 224, 224).data;
  const n = 224 * 224, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = (px[i * 4] / 255 - MEAN[0]) / STD[0];
    out[n + i] = (px[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    out[2 * n + i] = (px[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  const res = await session.run({ x: new ort.Tensor('float32', out, [1, 3, 224, 224]) });
  return l2(Float32Array.from(res.f.data));
}

// ------------------------------------------------------------ 사진 읽기
// JPEG 머리 부분만 읽어 크기·회전 정보 확인 (전체를 풀지 않아 빠름)
export async function jpegInfo(file) {
  try {
    const buf = new DataView(await file.slice(0, 256 * 1024).arrayBuffer());
    if (buf.getUint16(0) !== 0xFFD8) return null;
    let o = 2, ori = 1, w = 0, h = 0;
    while (o + 4 < buf.byteLength) {
      if (buf.getUint8(o) !== 0xFF) { o++; continue; }
      const m = buf.getUint8(o + 1), len = buf.getUint16(o + 2);
      if (m === 0xE1 && buf.getUint32(o + 4) === 0x45786966) {           // Exif
        const t = o + 10, le = buf.getUint16(t) === 0x4949;
        const g16 = p => buf.getUint16(p, le), g32 = p => buf.getUint32(p, le);
        const ifd = t + g32(t + 4), n = g16(ifd);
        for (let i = 0; i < n; i++) { const e = ifd + 2 + i * 12; if (g16(e) === 0x0112) ori = g16(e + 8); }
      } else if ((m >= 0xC0 && m <= 0xC3) || (m >= 0xC5 && m <= 0xC7) || (m >= 0xC9 && m <= 0xCB) || (m >= 0xCD && m <= 0xCF)) {
        h = buf.getUint16(o + 5); w = buf.getUint16(o + 7); break;
      }
      o += 2 + len;
    }
    if (!w || !h) return null;
    const rot = ori >= 5 && ori <= 8;
    return { w: rot ? h : w, h: rot ? w : h, ori };
  } catch (e) { return null; }
}

// 회전 바로잡은 비트맵을 원하는 크기로 (휴대폰 사진 회전 정보 반영)
export async function decodeTo(file, info, tw, th) {
  if (info) return await createImageBitmap(file, { imageOrientation: 'from-image', resizeWidth: tw, resizeHeight: th, resizeQuality: 'high' });
  const full = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const c = new OffscreenCanvas(tw, th); const g = c.getContext('2d');
  g.imageSmoothingQuality = 'high'; g.drawImage(full, 0, 0, tw, th); full.close();
  return c.transferToImageBitmap();
}

// 새 사진 1장: (1) 회전 바로잡아 작게 풀기 — 여러 장 동시에 가능
export async function preparePhoto(file) {
  let info = await jpegInfo(file);
  let w, h;
  if (info) { w = info.w; h = info.h; }
  else { const b = await createImageBitmap(file, { imageOrientation: 'from-image' }); w = b.width; h = b.height; b.close(); }
  const sc = 256 / Math.min(w, h);
  const tw = Math.max(224, Math.round(w * sc)), th = Math.max(224, Math.round(h * sc));
  const bmp = await decodeTo(file, info, tw, th);
  const cv = new OffscreenCanvas(tw, th); cv.getContext('2d').drawImage(bmp, 0, 0); bmp.close();
  return { cv, w, h, aspect: w / h, info };
}
// (2) 특징값 + 썸네일 — 한 장씩
export async function finishPhoto(prep) {
  const emb = await embedCanvas(prep.cv);
  const thumb = URL.createObjectURL(await prep.cv.convertToBlob({ type: 'image/jpeg', quality: 0.78 }));
  prep.cv = null;
  return { w: prep.w, h: prep.h, aspect: prep.aspect, info: prep.info, emb, thumb };
}

// 파일 이름에서 찍은 시각 (20260624_133959 / 13자리 ms) → 없으면 파일 수정 시각
export function shotTime(file) {
  const n = file.name.replace(/\.[^.]+$/, '');
  let m = n.match(/(20\d{2})(\d{2})(\d{2})[_\-\s]?(\d{2})(\d{2})(\d{2})/);
  if (m) return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
  m = n.match(/^(1[5-9]\d{11})$/);
  if (m) return +m[1] + 9 * 3600e3;
  return file.lastModified || 0;
}

// ------------------------------------------------------------ 폴더 → 건물 자동 연결
export function normName(s) {
  s = s.toLowerCase().replace(/20\d\d/g, '').replace('성능점검 현황표', '');
  s = s.replace(/new\s*no/g, 'newno').replace(/구\)\s*no/g, '구no').replace(/구\s+no/g, '구no');
  s = s.replace(/no\.?\s*/g, 'no');
  s = s.replace('lldpe', 'lld').replace('출하장', '출하동').replace('수지물류', '수지물류동').replace('동동', '동');
  s = s.replace('도헌정', '도헌정회관').replace('회관회관', '회관');
  s = s.replace('창고', '자동창고').replace('자동자동', '자동');
  s = s.replace(/154k?/g, '154');
  s = s.replace(/^\s*154\s*2단지/, '2단지154전기실').replace(/^\s*154\s*1단지/, '154전기실');
  s = s.replace(/[\s_\-()]/g, '');
  return s;
}
function bigrams(s) { const o = new Map(); for (let i = 0; i < s.length - 1; i++) { const k = s.slice(i, i + 2); o.set(k, (o.get(k) || 0) + 1); } return o; }
function dice(a, b) {
  if (a === b) return 1; if (a.length < 2 || b.length < 2) return 0;
  const A = bigrams(a), B = bigrams(b); let inter = 0, n = 0;
  for (const [k, v] of A) { inter += Math.min(v, B.get(k) || 0); n += v; } for (const v of B.values()) n += v;
  return 2 * inter / n;
}
export function autoMap(folders, buildings) {
  const out = {};
  for (const f of folders) {
    const nf = normName(f); let best = null;
    for (const b of buildings) {
      const r = dice(nf, normName(b.name));
      if (!best || r > best.score) best = { key: b.key, score: r };
    }
    out[f] = best && best.score >= 0.999 ? { key: best.key, sure: true } : { key: null, sure: false, hint: best && best.score >= 0.45 ? best.key : null };
  }
  return out;
}
export function normFac(s) { return (s || '').replace(/\s/g, '').replace('성능점검표', '').replace('현황표', ''); }

// ------------------------------------------------------------ 칸 안 자동 배열
export function bestGrid(aspects, W, H, gap = 100) {
  const n = aspects.length; let best = null;
  for (let cols = 1; cols <= Math.min(n, 4); cols++) {
    const rows = Math.ceil(n / cols); if (rows > 4) continue;
    const tw = W / cols, th = H / rows; let area = 0;
    for (const a of aspects) { const w = Math.min(tw - gap, (th - gap) * a); area += w * (w / a); }
    const score = area - (cols * rows - n) * 0.02 * W * H;
    if (!best || score > best.score) best = { score, cols, rows, tw, th };
  }
  const sizes = aspects.map(a => { const w = Math.min(best.tw - gap, (best.th - gap) * a); return [w, w / a]; });
  return { cols: best.cols, rows: best.rows, tw: best.tw, th: best.th, sizes };
}

// ------------------------------------------------------------ 워드 읽기
const kids = (el, name) => Array.from(el.children).filter(c => c.namespaceURI === NS.w && c.localName === name);
function findCells(doc) {
  const body = doc.getElementsByTagNameNS(NS.w, 'body')[0];
  let bld = null, fac = null; const out = [];
  for (const el of Array.from(body.children)) {
    if (el.namespaceURI !== NS.w) continue;
    if (el.localName === 'p') {
      const t = el.textContent.trim();
      const m = t.match(/^1\.(\d+)\s+(.+)$/);
      if (m) { bld = { n: +m[1], name: m[2].trim(), el }; fac = null; continue; }
      if (bld && t.replace(/\s/g, '').includes('성능점검표')) fac = t.replace(/\s/g, '').replace('성능점검표', '');
    } else if (el.localName === 'tbl' && bld) {
      const tr = kids(el, 'tr')[0]; if (!tr) continue;
      const tcs = kids(tr, 'tc');
      if (tcs.length === 2 && tcs[0].textContent.replace(/\s/g, '') === '현황사진')
        out.push({ tbl: el, tr, tc: tcs[1], b: bld, f: fac || '설비' });
    }
  }
  return out;
}
function cellBox(c) {
  const tcW = c.tc.getElementsByTagNameNS(NS.w, 'tcW')[0];
  const trH = c.tr.getElementsByTagNameNS(NS.w, 'trHeight')[0];
  const w = tcW ? +tcW.getAttributeNS(NS.w, 'w') : 8129;
  const h = trH ? +trH.getAttributeNS(NS.w, 'val') : 10771;
  return { W: w - 216, H: h - 470 };
}
const ser = d => { let s = new XMLSerializer().serializeToString(d); s = s.replace(/^<\?xml[^>]*\?>\s*/, ''); return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n' + s; };

// 기존 사진(워드에 이미 들어 있는 사진)을 돌림·뒤집기까지 반영해 그림으로
async function orientedFromMedia(blob, rot, flipH, flipV) {
  const b = await createImageBitmap(blob);
  const q = ((Math.round(rot / 5400000) % 4) + 4) % 4;
  const sw = q % 2 ? b.height : b.width, sh = q % 2 ? b.width : b.height;
  const sc = Math.min(1, 1800 / Math.max(sw, sh));
  const W = Math.round(sw * sc), H = Math.round(sh * sc);
  const c = new OffscreenCanvas(W, H), g = c.getContext('2d');
  g.translate(W / 2, H / 2); g.rotate(q * Math.PI / 2); g.scale(flipH ? -1 : 1, flipV ? -1 : 1);
  const dw = q % 2 ? H : W, dh = q % 2 ? W : H;
  g.drawImage(b, -dw / 2, -dh / 2, dw, dh); b.close();
  return c;
}

export async function readDoc(file, onStep) {
  onStep && onStep('워드 파일 여는 중');
  const zip = await JSZip.loadAsync(file);
  const xmlStr = await zip.file('word/document.xml').async('string');
  onStep && onStep('워드 내용 분석 중 (문서가 크면 10초 정도)');
  await new Promise(r => setTimeout(r, 30));
  const doc = new DOMParser().parseFromString(xmlStr, 'application/xml');
  if (doc.getElementsByTagName('parsererror').length) throw new Error('워드 내용을 읽지 못했습니다 (손상된 파일일 수 있음)');
  const relsDoc = new DOMParser().parseFromString(await zip.file('word/_rels/document.xml.rels').async('string'), 'application/xml');
  const rels = {};
  for (const r of Array.from(relsDoc.getElementsByTagNameNS(NS.rel, 'Relationship'))) rels[r.getAttribute('Id')] = r.getAttribute('Target');
  const found = findCells(doc);
  const cells = [], buildings = [], olds = [];
  const bmap = new Map();
  for (let i = 0; i < found.length; i++) {
    const c = found[i];
    const key = c.b.n + '|' + c.b.name;
    if (!bmap.has(key)) { const b = { key, n: c.b.n, name: c.b.name, cells: [] }; bmap.set(key, b); buildings.push(b); }
    const cell = { i, bkey: key, f: c.f, fn: normFac(c.f), box: cellBox(c), olds: [], locked: false };
    for (const p of Array.from(c.tc.getElementsByTagNameNS(NS.pic, 'pic'))) {
      const blip = p.getElementsByTagNameNS(NS.a, 'blip')[0];
      const rid = blip && blip.getAttributeNS(NS.r, 'embed');
      const xf = p.getElementsByTagNameNS(NS.a, 'xfrm')[0];
      const o = { id: 'o' + olds.length, cell: i, rid, target: rels[rid],
        rot: xf ? +(xf.getAttribute('rot') || 0) : 0, flipH: xf && xf.getAttribute('flipH') === '1', flipV: xf && xf.getAttribute('flipV') === '1' };
      olds.push(o); cell.olds.push(o.id);
    }
    bmap.get(key).cells.push(i); cells.push(cell);
  }
  const anchors = doc.getElementsByTagNameNS(NS.wp, 'anchor').length;
  // 건물별 본문 범위 (워드 미리보기를 건물 단위로 빠르게 그리기 위해)
  const body = doc.getElementsByTagNameNS(NS.w, 'body')[0];
  const bodyKids = Array.from(body.children);
  const pos = new Map(bodyKids.map((el, i) => [el, i]));
  const heads = found.reduce((m, c) => { const k = c.b.n + '|' + c.b.name; if (!m.has(k)) m.set(k, c.b.el); return m; }, new Map());
  const last = bodyKids[bodyKids.length - 1];
  const endAll = last && last.localName === 'sectPr' ? bodyKids.length - 1 : bodyKids.length;
  buildings.forEach((b, i) => { b.start = pos.get(heads.get(b.key)); });
  buildings.forEach((b, i) => { b.end = i + 1 < buildings.length ? buildings[i + 1].start : endAll; });
  const mb = xmlStr.match(/<w:body[^>]*>/);
  const xmlHead = xmlStr.slice(0, mb.index + mb[0].length);
  const sectPrXml = last && last.localName === 'sectPr' ? new XMLSerializer().serializeToString(last) : '';
  return { zip, doc, bodyKids, xmlHead, sectPrXml, rels, xmlLen: xmlStr.length, cells, buildings, olds, anchors, fileName: file.name };
}

// 기존 사진들: 썸네일·특징값(설비 종류 학습용)
export async function analyzeOlds(D, onEach) {
  for (const o of D.olds) {
    try {
      const f = o.target && D.zip.file('word/' + o.target.replace(/^\/?word\//, ''));
      if (!f) throw new Error('missing');
      const blob = await f.async('blob');
      const cv = await orientedFromMedia(blob, o.rot, o.flipH, o.flipV);
      o.w = cv.width; o.h = cv.height; o.aspect = cv.width / cv.height;
      o.full = await cv.convertToBlob({ type: 'image/jpeg', quality: 0.9 });
      const sc = 256 / Math.min(cv.width, cv.height);
      const tw = Math.max(224, Math.round(cv.width * sc)), th = Math.max(224, Math.round(cv.height * sc));
      const small = new OffscreenCanvas(tw, th); small.getContext('2d').drawImage(cv, 0, 0, tw, th);
      o.emb = await embedCanvas(small);
      o.thumb = URL.createObjectURL(await small.convertToBlob({ type: 'image/jpeg', quality: 0.78 }));
    } catch (e) {
      o.bad = true; D.cells[o.cell].locked = true;    // 읽을 수 없는 그림이 있는 칸은 그대로 둠
    }
    onEach && onEach(o);
  }
}

// ------------------------------------------------------------ 건물 하나만 담은 작은 워드 (미리보기용, 원본은 그대로)
export async function miniDocx(D, b) {
  const xs = new XMLSerializer();
  let body = '';
  for (let i = b.start; i < b.end; i++) body += xs.serializeToString(D.bodyKids[i]);
  const xml = D.xmlHead + body + D.sectPrXml + '</w:body></w:document>';
  const used = new Set(); for (const m of xml.matchAll(/r:(?:embed|id|link|pict)="([^"]+)"/g)) used.add(m[1]);
  const mini = new JSZip();
  for (const n of Object.keys(D.zip.files)) {
    const f = D.zip.files[n]; if (f.dir || n === 'word/document.xml' || /^word\/(media|embeddings)\//.test(n)) continue;
    mini.file(n, await f.async('uint8array'), { createFolders: false });
  }
  for (const id of used) {
    const t = D.rels[id]; if (!t || /^https?:/.test(t)) continue;
    const p = 'word/' + t.replace(/^\/?word\//, '');
    const f = D.zip.file(p); if (f) mini.file(p, await f.async('uint8array'), { createFolders: false });
  }
  mini.file('word/document.xml', xml, { createFolders: false });
  return await mini.generateAsync({ type: 'blob', compression: 'STORE' });
}

// ------------------------------------------------------------ 워드 만들기
const PIC_XML = (cx, cy, id, name, rid) =>
  `<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="${cx}" cy="${cy}"/><wp:docPr id="${id}" name="sitelog-photo" descr="${esc(name)}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="${esc(name)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="${rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="${cx}" cy="${cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>`;
function esc(s) { return String(s).replace(/[<>&"]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c])); }

function cellXml(items, box) {
  if (!items.length) return { xml: '<w:p><w:pPr><w:pStyle w:val="TableParagraph"/></w:pPr></w:p>' };
  const g = bestGrid(items.map(it => it.aspect), box.W, box.H, 100);
  const TW = Math.floor(g.tw) * g.cols;
  const SP = '<w:r><w:rPr><w:sz w:val="16"/></w:rPr><w:t xml:space="preserve">  </w:t></w:r>';
  let x = `<w:tbl><w:tblPr><w:tblW w:w="${TW}" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar><w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders><w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/></w:tblPr><w:tblGrid><w:gridCol w:w="${TW}"/></w:tblGrid>`;
  let k = 0;
  for (let r = 0; r < g.rows; r++) {
    const inrow = Math.min(g.cols, items.length - k);
    let rh = 0; for (let c = 0; c < inrow; c++) rh = Math.max(rh, g.sizes[k + c][1]);
    x += `<w:tr><w:trPr><w:cantSplit/><w:trHeight w:val="${Math.round(rh + 110)}" w:hRule="exact"/></w:trPr><w:tc><w:tcPr><w:tcW w:w="${TW}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>`;
    for (let c = 0; c < inrow; c++) {
      const it = items[k], [w, h] = g.sizes[k];
      it.tw = w; it.th = h;
      if (c) x += SP;
      x += PIC_XML(Math.round(w * 635), Math.round(h * 635), it.docPrId, it.name, it.rid);
      k++;
    }
    x += '</w:p></w:tc></w:tr>';
  }
  x += '</w:tbl><w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="20" w:lineRule="exact"/></w:pPr></w:p>';
  return { xml: x, grid: g };
}
// 1) 배열 계산으로 사진 크기 확정 → 2) 그 크기에 맞춰 사진 압축
function wrapNs(inner) {
  return `<root xmlns:w="${NS.w}" xmlns:wp="${NS.wp}" xmlns:a="${NS.a}" xmlns:pic="${NS.pic}" xmlns:r="${NS.r}">${inner}</root>`;
}

export async function buildDocx(D, plan, getSource, onProgress) {
  // plan: [{cell, items:[{key,name,aspect}]}] — 바뀐 칸만
  const doc = D.doc, zip = D.zip;
  const cellsNow = findCells(doc);
  if (cellsNow.length !== D.cells.length) throw new Error('워드 구조가 처음 읽었을 때와 다릅니다. 처음부터 다시 해 주세요.');
  const relsPath = 'word/_rels/document.xml.rels';
  const relsDoc = new DOMParser().parseFromString(await zip.file(relsPath).async('string'), 'application/xml');
  const relRoot = relsDoc.documentElement;
  const usedIds = new Set(Array.from(relRoot.getElementsByTagNameNS(NS.rel, 'Relationship')).map(r => r.getAttribute('Id')));
  let ridN = 1, maxDoc = 0;
  for (const d of Array.from(doc.getElementsByTagNameNS(NS.wp, 'docPr'))) maxDoc = Math.max(maxDoc, +d.getAttribute('id') || 0);
  let docPrId = Math.max(maxDoc + 1, 7000000);
  const newRid = () => { let id; do { id = 'rIdSL' + (ridN++); } while (usedIds.has(id)); usedIds.add(id); return id; };

  const saved = [];          // 원래 내용 보관 → 저장 후 되돌림 (다시 수정·재생성 가능하게)
  const media = [];
  const total = plan.reduce((s, p) => s + p.items.length, 0); let done = 0;
  const outName = D.fileName.replace(/\.docx$/i, '') + '_사진완성.docx';
  const newMedia = [];
  try {
    // (1) 칸마다 배열 계산 → (2) 사진을 그 크기로 동시에 여러 장 압축 → (3) 워드에 넣기
    const jobs = [], built = [];
    for (const p of plan) {
      const items = p.items.map(it => ({ ...it, rid: newRid(), docPrId: docPrId++ }));
      const { xml } = cellXml(items, D.cells[p.cell].box);
      built.push({ p, items, xml });
      for (const it of items) jobs.push(it);
    }
    let next = 0;
    const worker = async () => {
      while (next < jobs.length) {
        const it = jobs[next++];
        const px = [Math.max(16, Math.round(it.tw / 1440 * DPI)), Math.max(16, Math.round(it.th / 1440 * DPI))];
        it.blob = await getSource(it.key, px[0], px[1]);
        done++; onProgress && onProgress(done, total, '사진 크기 맞춰 줄이는 중');
      }
    };
    await Promise.all([worker(), worker(), worker(), worker()]);
    for (const { p, items, xml } of built) {
      const tc = cellsNow[p.cell].tc;
      const frag = new DOMParser().parseFromString(wrapNs(xml), 'application/xml');
      if (frag.getElementsByTagName('parsererror').length) throw new Error('사진 칸 XML 생성 오류');
      const old = Array.from(tc.childNodes).filter(n => !(n.nodeType === 1 && n.localName === 'tcPr'));
      saved.push({ tc, old });
      old.forEach(n => tc.removeChild(n));
      for (const n of Array.from(frag.documentElement.childNodes)) tc.appendChild(doc.importNode(n, true));
      for (const it of items) {
        const fn = `media/sitelog_${it.docPrId}.jpeg`;
        newMedia.push(['word/' + fn, it.blob]); it.blob = null;
        const rel = relsDoc.createElementNS(NS.rel, 'Relationship');
        rel.setAttribute('Id', it.rid);
        rel.setAttribute('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image');
        rel.setAttribute('Target', fn);
        relRoot.appendChild(rel);
      }
    }
    // 검사: 계통도·도형 그대로인지, 넣은 사진 수가 맞는지
    onProgress && onProgress(total, total, '검사 중');
    const anchorsAfter = doc.getElementsByTagNameNS(NS.wp, 'anchor').length;
    if (anchorsAfter !== D.anchors) throw new Error(`검사 실패: 도형/계통도 수가 바뀜 (${D.anchors} → ${anchorsAfter})`);
    const check = findCells(doc);
    if (check.length !== D.cells.length) throw new Error('검사 실패: 현황사진 칸 수가 바뀜');
    let placed = 0;
    for (const p of plan) {
      const n = Array.from(check[p.cell].tc.getElementsByTagNameNS(NS.wp, 'docPr')).filter(d => d.getAttribute('name') === 'sitelog-photo').length;
      if (n !== p.items.length) throw new Error(`검사 실패: ${D.cells[p.cell].f} 칸 사진 수 불일치 (${n}/${p.items.length})`);
      placed += n;
    }
    onProgress && onProgress(total, total, '워드 파일로 저장 중');
    await new Promise(r => setTimeout(r, 30));
    const xmlOut = ser(doc);
    const out = new JSZip();
    // 원본 파일들 그대로 복사 (document.xml, rels, content types만 바꿈)
    const names = Object.keys(zip.files).filter(n => !zip.files[n].dir);
    for (const n of names) {
      if (n === 'word/document.xml' || n === relsPath || n === '[Content_Types].xml') continue;
      const media = /\.(jpe?g|png|gif)$/i.test(n);   // 이미 압축된 그림은 다시 압축하지 않음(빠름)
      out.file(n, await zip.file(n).async('uint8array'), media ? { createFolders: false, compression: 'STORE' } : { createFolders: false });
    }
    out.file('word/document.xml', xmlOut, { createFolders: false });
    out.file(relsPath, ser(relsDoc), { createFolders: false });
    let ct = await zip.file('[Content_Types].xml').async('string');
    if (!/Extension="jpeg"/i.test(ct)) ct = ct.replace('</Types>', '<Default Extension="jpeg" ContentType="image/jpeg"/></Types>');
    out.file('[Content_Types].xml', ct, { createFolders: false });
    for (const [n, b] of newMedia) out.file(n, b, { binary: true, compression: 'STORE', createFolders: false });
    for (const n of Object.keys(out.files)) if (out.files[n].dir) delete out.files[n];
    const blob = await out.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 6 },
      mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' },
      m => onProgress && onProgress(total, total, `워드 파일로 묶는 중 ${Math.round(m.percent)}%`));
    return { blob, name: outName, placed, cellsChanged: plan.length, mediaBytes: newMedia.reduce((s, [, b]) => s + b.size, 0) };
  } finally {
    // 원래 내용 되돌리기 → 다시 수정해서 또 만들 수 있음
    for (const { tc, old } of saved) {
      Array.from(tc.childNodes).filter(n => !(n.nodeType === 1 && n.localName === 'tcPr')).forEach(n => tc.removeChild(n));
      old.forEach(n => tc.appendChild(n));
    }
    // rels 문서는 매번 zip에서 다시 읽으므로 되돌릴 필요 없음
  }
}

// 새 사진 → 정해진 픽셀 크기 JPEG
export async function photoJpeg(file, info, pw, ph, aspect) {
  // 비율 유지: 표시 크기(pw×ph)는 이미 사진 비율과 같음
  const bmp = await decodeTo(file, info, pw, ph);
  const c = new OffscreenCanvas(pw, ph); c.getContext('2d').drawImage(bmp, 0, 0); bmp.close();
  return await c.convertToBlob({ type: 'image/jpeg', quality: JPEG_Q });
}
export async function oldJpeg(blob, pw, ph) {
  const b = await createImageBitmap(blob);
  const c = new OffscreenCanvas(pw, ph), g = c.getContext('2d'); g.imageSmoothingQuality = 'high';
  g.drawImage(b, 0, 0, pw, ph); b.close();
  return await c.convertToBlob({ type: 'image/jpeg', quality: JPEG_Q });
}
