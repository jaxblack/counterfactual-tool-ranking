import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import MarkdownIt from 'markdown-it';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(directory);
const destination = path.join(root, 'submission');
const markdown = await fs.readFile(path.join(directory, 'paper.md'), 'utf8');
const tokens = new MarkdownIt().parse(markdown, {});
const titleIndex = tokens.findIndex(token => token.type === 'heading_open' && token.tag === 'h1');
assert.ok(titleIndex >= 0, 'Paper title is missing');
const title = tokens[titleIndex + 1].content;
let collecting = false;
const abstractParagraphs = [];
for (let index = 0; index < tokens.length; index++) {
  const token = tokens[index];
  if (token.type === 'heading_open') {
    if (collecting) break;
    collecting = tokens[index + 1]?.content === 'Abstract';
  } else if (collecting && token.type === 'paragraph_open') {
    const inline = tokens[index + 1];
    if (inline.content.startsWith('**Keywords:')) break;
    abstractParagraphs.push(inline.content.replace(/\s+/g, ' ').trim());
  }
}
const abstract = abstractParagraphs.join('\n\n');
assert.ok(abstract.length > 500 && abstract.length < 5000, 'Unexpected abstract length');
assert.ok(!markdown.includes('{{'), 'Paper has unrendered claims');
assert.ok(markdown.includes('**jiapengli** (Microsoft)'), 'Author block changed; confirm metadata');
assert.ok(markdown.includes('## Generative-AI Assistance'), 'AI-use disclosure is missing');
const pdf = await fs.readFile(path.join(directory, 'paper.pdf'));
assert.equal(pdf.subarray(0, 5).toString(), '%PDF-');
const digest = createHash('sha256').update(pdf).digest('hex');
const fileName = 'counterfactual-tool-ranking.pdf';
const metadata = {
  title,
  authors: [{ name: 'jiapengli', affiliation: 'Microsoft' }],
  abstract,
  primary_category_proposed: 'cs.LG',
  comments: 'Working paper. Synthetic single-decision tasks; code and artifacts are publicly available.',
  journal_reference: null,
  doi: null,
  license: null,
};
const status = {
  status: 'prepared_not_submitted',
  submission_id: null,
  arxiv_id: null,
  upload_file: fileName,
  sha256: digest,
  pdf_source: 'Markdown and HTML rendered by Chromium; not generated from TeX',
  entrypoint: 'https://arxiv.org/user',
  registration: 'https://arxiv.org/user/register',
  pending: [
    'Author creates and verifies their own arXiv account.',
    'Author confirms intended spelling of their publication name and final author list.',
    'Author selects a distribution license and confirms the right to grant it.',
    'Author reviews the paper, AI disclosure and arXiv submission agreement.',
    'Complete subject-area endorsement if requested by arXiv.',
    'Review the processed PDF and metadata, then explicitly submit through the author account.',
  ],
};
await fs.mkdir(destination, { recursive: true });
await fs.writeFile(path.join(destination, fileName), pdf);
await fs.writeFile(path.join(destination, 'abstract.txt'), abstract + '\n');
await fs.writeFile(path.join(destination, 'metadata.json'), JSON.stringify(metadata, null, 2) + '\n');
await fs.writeFile(path.join(destination, 'status.json'), JSON.stringify(status, null, 2) + '\n');
console.log(JSON.stringify({ status: status.status, pdf: fileName, bytes: pdf.length, sha256: digest, abstract_characters: abstract.length }, null, 2));