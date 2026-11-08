// 감독일지 사진 자동 첨부 — 브라우저 엔진 (사진은 이 PC 밖으로 나가지 않음)
import * as ort from './ort.wasm.bundle.min.mjs';

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
let session = null;

// ------------------------------------------------------------ 다운로드(진행률)
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

// 모델은 호스팅 제약 때문에 base64 JSON 2조각으로 나눠 둠 → 받아서 다시 합침
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
  const tick = () => { const g = parts.wasm[0] + parts.model[0], t = parts.wasm[1] + parts.model[1]; onProgress && onProgress(g, t); };
  const [wasm, model] = await Promise.all([
    fetchWithProgress('./ort-wasm-simd-threaded.wasm', (g, t) => { parts.wasm = [g, t || parts.wasm[1]]; tick(); }),
    fetchModel((g, t) => { parts.model = [g, t]; tick(); }),
  ]);
  ort.env.wasm.wasmBinary = wasm;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: ['wasm'] });
}

// ------------------------------------------------------------ 이미지
export async function decode(file) {
  // EXIF 회전 반영해서 비트맵으로
  return await createImageBitmap(file, { imageOrientation: 'from-image' });
}

function toTensorData(src, sx, sy, sw, sh, mirror) {
  // 짧은 변 256으로 줄인 뒤 가운데 224 자르기 (torchvision 기본 전처리와 같은 방식)
  const sc = 256 / Math.min(sw, sh);
  const rw = Math.round(sw * sc), rh = Math.round(sh * sc);
  const c1 = new OffscreenCanvas(rw, rh), g1 = c1.getContext('2d');
  g1.imageSmoothingEnabled = true; g1.imageSmoothingQuality = 'high';
  if (mirror) { g1.translate(rw, 0); g1.scale(-1, 1); }
  g1.drawImage(src, sx, sy, sw, sh, 0, 0, rw, rh);
  const ox = Math.floor((rw - 224) / 2), oy = Math.floor((rh - 224) / 2);
  const px = g1.getImageData(ox, oy, 224, 224).data;
  const out = new Float32Array(3 * 224 * 224), n = 224 * 224;
  for (let i = 0; i < n; i++) {
    out[i] = (px[i * 4] / 255 - MEAN[0]) / STD[0];
    out[n + i] = (px[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    out[2 * n + i] = (px[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return out;
}

function l2(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map(x => x / s); }

export async function embed(bitmap) {
  const w = bitmap.width, h = bitmap.height;
  const views = [
    toTensorData(bitmap, 0, 0, w, h, false),
    toTensorData(bitmap, 0, 0, w, h, true),
    toTensorData(bitmap, w * .12, h * .12, w * .76, h * .76, false),
  ];
  const batch = new Float32Array(3 * 3 * 224 * 224);
  views.forEach((v, i) => batch.set(v, i * v.length));
  const out = await session.run({ x: new ort.Tensor('float32', batch, [3, 3, 224, 224]) });
  const f = out.f.data, D = f.length / 3;
  const acc = new Float32Array(D);
  for (let i = 0; i < 3; i++) { const v = l2(f.slice(i * D, (i + 1) * D)); for (let j = 0; j < D; j++) acc[j] += v[j]; }
  return l2(acc);
}

// ------------------------------------------------------------ 분류
export function b64ToF32(b64) {
  const s = atob(b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
}
export function f32ToB64(f) {
  const u = new Uint8Array(f.buffer.slice(0)); let s = '';
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
  return btoa(s);
}

export function classify(emb, refs, minSim = 0.45, minMargin = 0.03) {
  const best = {};
  for (const r of refs) {
    let s = 0; for (let j = 0; j < emb.length; j++) s += emb[j] * r.emb[j];
    if (best[r.k] === undefined || s > best[r.k]) best[r.k] = s;
  }
  const order = Object.entries(best).sort((a, b) => b[1] - a[1]);
  const [k1, s1] = order[0], [k2, s2] = order[1] || [0, -1];
  return { section: +k1, sim: +s1.toFixed(3), second: +k2, margin: +(s1 - s2).toFixed(3),
           status: (s1 >= minSim && s1 - s2 >= minMargin) ? 'ok' : 'check' };
}

// 섹션별 사진 칸(테두리 안) 크기. 2칸 배열은 원본 완성본 크기, 3칸 배열은 정사각형
const TOTAL_TW = 8926;
export function boxFor(layout) {
  const L = layout || { cols: 2, box_w: 7.6 * EMU_CM, box_h: 7.6 * EMU_CM };
  const cw = Math.floor(TOTAL_TW / L.cols);
  const w = cw * 635 - Math.round(0.4 * EMU_CM);   // 칸 폭 - 좌우 여백
  const h = L.cols >= 3 ? w : Math.round(L.box_h);
  return { cols: L.cols, cw, w, h };
}

// 워드에 넣을 사진: 긴 변 1400px JPEG
// aspect(가로/세로)를 주면 칸 비율에 맞게 가운데를 잘라 칸을 꽉 채움
export async function prepJpeg(bitmap, maxPx = 1400, aspect = null) {
  let sx = 0, sy = 0, sw = bitmap.width, sh = bitmap.height;
  if (aspect) {
    if (sw / sh > aspect) { const nw = sh * aspect; sx = (sw - nw) / 2; sw = nw; }
    else { const nh = sw / aspect; sy = (sh - nh) / 2; sh = nh; }
  }
  const sc = Math.min(1, maxPx / Math.max(sw, sh));
  const w = Math.round(sw * sc), h = Math.round(sh * sc);
  const c = new OffscreenCanvas(w, h); const g = c.getContext('2d');
  g.imageSmoothingQuality = 'high'; g.drawImage(bitmap, sx, sy, sw, sh, 0, 0, w, h);
  const blob = await c.convertToBlob({ type: 'image/jpeg', quality: 0.85 });
  return { data: new Uint8Array(await blob.arrayBuffer()), w, h };
}

// ------------------------------------------------------------ 워드(docx)
const NS = {
  w: 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
  r: 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
  wp: 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
  a: 'http://schemas.openxmlformats.org/drawingml/2006/main',
  pic: 'http://schemas.openxmlformats.org/drawingml/2006/picture',
  pr: 'http://schemas.openxmlformats.org/package/2006/relationships',
  ct: 'http://schemas.openxmlformats.org/package/2006/content-types',
};
const SEC_RE = /^\s*1\s*[–\-]\s*(\d+)\s*\.\s*(.+?)\s*$/;
const DIAG_KEY = '설비 계통 상세도', REPAIR_KEY = '보수 및 교체';
const EMU_CM = 360000;

const kids = (el, local) => Array.from(el.childNodes).filter(n => n.nodeType === 1 && n.namespaceURI === NS.w && (!local || n.localName === local));
const text = el => Array.from(el.getElementsByTagNameNS(NS.w, 't')).map(t => t.textContent).join('').trim();
const parse = s => new DOMParser().parseFromString(s, 'application/xml');
const xmlHeader = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n';
// 브라우저마다 직렬화 결과에 XML 선언이 있을 수도 없을 수도 있음 → 지우고 하나만 붙임
const ser = node => xmlHeader + new XMLSerializer().serializeToString(node).replace(/^\s*<\?xml[^>]*\?>\s*/, '');

function findSections(body) {
  const els = kids(body);
  const heads = [];
  els.forEach((el, i) => {
    if (el.localName !== 'p') return;
    const t = text(el); const m = t.match(SEC_RE);
    if (m && t.length < 60) heads.push({ k: +m[1], title: m[2], i });
  });
  return heads.map((h, n) => {
    let end = n + 1 < heads.length ? heads[n + 1].i : els.length;
    if (n === heads.length - 1) {
      for (let j = h.i; j < end; j++) if (els[j].localName === 'tbl' && text(els[j]).includes(REPAIR_KEY)) { end = j + 1; break; }
    }
    return { ...h, els: els.slice(h.i, end) };
  });
}

function splitTableAt(tbl, tr) {
  const trs = kids(tbl, 'tr'); const i = trs.indexOf(tr);
  if (i === 0) return tbl;
  const doc = tbl.ownerDocument;
  const nt = doc.createElementNS(NS.w, 'w:tbl');
  nt.appendChild(kids(tbl, 'tblPr')[0].cloneNode(true));
  nt.appendChild(kids(tbl, 'tblGrid')[0].cloneNode(true));
  for (const t of trs.slice(i)) nt.appendChild(t);
  tbl.parentNode.insertBefore(nt, tbl.nextSibling);
  return nt;
}

function keepTogether(tbl) {
  const doc = tbl.ownerDocument; const trs = kids(tbl, 'tr');
  trs.forEach((tr, i) => {
    let trPr = kids(tr, 'trPr')[0];
    if (!trPr) { trPr = doc.createElementNS(NS.w, 'w:trPr'); tr.insertBefore(trPr, tr.firstChild); }
    if (!kids(trPr, 'cantSplit').length) trPr.insertBefore(doc.createElementNS(NS.w, 'w:cantSplit'), trPr.firstChild);
    if (i < trs.length - 1) for (const p of tr.getElementsByTagNameNS(NS.w, 'p')) {
      let pPr = kids(p, 'pPr')[0];
      if (!pPr) { pPr = doc.createElementNS(NS.w, 'w:pPr'); p.insertBefore(pPr, p.firstChild); }
      if (!kids(pPr, 'keepNext').length) pPr.insertBefore(doc.createElementNS(NS.w, 'w:keepNext'), pPr.firstChild);
    }
  });
}

class Pkg {
  constructor(zip, rels, ct) { this.zip = zip; this.rels = rels; this.ct = ct; this.n = 0; }
  static async open(zip) {
    const rels = parse(await zip.file('word/_rels/document.xml.rels').async('string'));
    const ct = parse(await zip.file('[Content_Types].xml').async('string'));
    return new Pkg(zip, rels, ct);
  }
  addImage(bytes, ext) {
    ext = ext.toLowerCase(); if (ext === 'jpeg') ext = 'jpg';
    const name = `sitelog_${Date.now().toString(36)}_${++this.n}.${ext}`;
    this.zip.file('word/media/' + name, bytes, { createFolders: false });
    const root = this.rels.documentElement;
    const ids = new Set(Array.from(root.getElementsByTagNameNS(NS.pr, 'Relationship')).map(r => r.getAttribute('Id')));
    let id; do { id = 'rIdS' + (++this.n); } while (ids.has(id));
    const r = this.rels.createElementNS(NS.pr, 'Relationship');
    r.setAttribute('Id', id);
    r.setAttribute('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image');
    r.setAttribute('Target', 'media/' + name);
    root.appendChild(r);
    const has = Array.from(this.ct.documentElement.getElementsByTagNameNS(NS.ct, 'Default')).some(d => d.getAttribute('Extension').toLowerCase() === ext);
    if (!has) {
      const d = this.ct.createElementNS(NS.ct, 'Default');
      d.setAttribute('Extension', ext); d.setAttribute('ContentType', ext === 'png' ? 'image/png' : 'image/jpeg');
      this.ct.documentElement.appendChild(d);
    }
    return id;
  }
  save() {
    this.zip.file('word/_rels/document.xml.rels', ser(this.rels), { createFolders: false });
    this.zip.file('[Content_Types].xml', ser(this.ct), { createFolders: false });
  }
}

function b64ToBytes(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }

function importDiagram(doc, pkg, d) {
  const el = doc.importNode(parse(d.xml).documentElement, true);
  const remap = {};
  for (const m of d.media) remap[m.rid] = pkg.addImage(b64ToBytes(m.b64), m.ext);
  for (const e of [el, ...el.getElementsByTagName('*')])
    for (const a of Array.from(e.attributes || []))
      if (a.namespaceURI === NS.r && remap[a.value]) e.setAttributeNS(NS.r, a.name, remap[a.value]);
  return el;
}

function photoTableXml(cells, box) {
  const { cols, cw, w: boxW, h: boxH } = box;
  const rows = [];
  for (let i = 0; i < Math.max(1, Math.ceil(cells.length / cols)); i++) {
    let r = '<w:tr><w:trPr><w:cantSplit/></w:trPr>';
    for (let c = 0; c < cols; c++) {
      const it = cells[i * cols + c];
      let run = '';
      if (it) {
        const sc = Math.min(boxW / it.w, boxH / it.h);
        const cx = Math.round(it.w * sc), cy = Math.round(it.h * sc);
        run = `<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="${cx}" cy="${cy}"/><wp:docPr id="1" name="사진"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="photo.jpg"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="${it.rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="${cx}" cy="${cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>`;
      }
      r += `<w:tc><w:tcPr><w:tcW w:w="${cw}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr>${i < Math.ceil(cells.length / cols) - 1 ? '<w:keepNext/>' : ''}<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:ind w:left="0" w:right="0" w:firstLine="0"/><w:jc w:val="center"/></w:pPr>${run}</w:p></w:tc>`;
    }
    rows.push(r + '</w:tr>');
  }
  const b = ['top', 'left', 'bottom', 'right', 'insideH', 'insideV'].map(x => `<w:${x} w:val="single" w:sz="4" w:space="0" w:color="000000"/>`).join('');
  return `<w:tbl xmlns:w="${NS.w}" xmlns:r="${NS.r}" xmlns:wp="${NS.wp}" xmlns:a="${NS.a}" xmlns:pic="${NS.pic}"><w:tblPr><w:tblW w:w="${cw * cols}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>${b}</w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>${('<w:gridCol w:w="' + cw + '"/>').repeat(cols)}</w:tblGrid>${rows.join('')}</w:tbl>`;
}

/**
 * blankFile: File(.docx)  lib: 라이브러리  assign: {k: [{data,w,h}]}  opts: {diagrams:true}
 * 반환: {blob, report}
 */
export async function assemble(blankFile, lib, assign, opts = {}) {
  const zip = await JSZip.loadAsync(blankFile, { createFolders: false });
  const doc = parse(await zip.file('word/document.xml').async('string'));
  const pkg = await Pkg.open(zip);
  const body = doc.getElementsByTagNameNS(NS.w, 'body')[0];
  const report = [];
  const secs = findSections(body);
  if (!secs.length) throw new Error("워드에서 '1 – 1. ...' 형식의 섹션 제목을 찾지 못했습니다.");
  for (const s of secs.reverse()) {
    const info = lib.sections[String(s.k)] || {};
    let anchor = null;
    outer: for (const el of s.els) {
      if (el.localName !== 'tbl') continue;
      for (const tr of kids(el, 'tr')) if (text(tr).startsWith(REPAIR_KEY)) { anchor = splitTableAt(el, tr); break outer; }
    }
    if (!anchor) { report.push({ k: s.k, title: s.title, n: 0, skipped: (assign[s.k] || []).length, msg: '조치내역 표를 못 찾아 건너뜀' }); continue; }
    const add = [];
    let diag = false;
    if (opts.diagrams !== false && info.diagram) { add.push(importDiagram(doc, pkg, info.diagram)); diag = true; }
    const photos = assign[s.k] || [];
    if (photos.length) {
      const cells = photos.map(p => ({ rid: pkg.addImage(p.data, 'jpg'), w: p.w, h: p.h }));
      add.push(doc.importNode(parse(photoTableXml(cells, boxFor(info.layout))).documentElement, true));
    }
    for (const el of add) {
      keepTogether(el);
      anchor.parentNode.insertBefore(el, anchor);
      anchor.parentNode.insertBefore(doc.createElementNS(NS.w, 'w:p'), anchor);
    }
    report.push({ k: s.k, title: s.title, n: photos.length, msg: `계통도 ${diag ? 'O' : 'X'} / 사진 ${photos.length}장` });
  }
  let id = 1;
  for (const dp of doc.getElementsByTagNameNS(NS.wp, 'docPr')) dp.setAttribute('id', String(id++));
  zip.file('word/document.xml', ser(doc), { createFolders: false });
  // 폴더 항목만 지움 (zip.remove는 하위 파일까지 지우므로 쓰지 않음)
  for (const n of Object.keys(zip.files)) if (zip.files[n].dir) delete zip.files[n];
  pkg.save();
  const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE',
    mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
  return { blob, report: report.sort((a, b) => a.k - b.k) };
}
