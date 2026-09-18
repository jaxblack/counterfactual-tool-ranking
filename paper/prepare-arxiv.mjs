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
assert.ok(markdown.includes('**Jiapeng Li** (Microsoft)'), 'Author block changed; confirm metadata');
assert.ok(markdown.includes('## Generative-AI Assistance'), 'AI-use disclosure is missing');
const pdf = await fs.readFile(path.join(directory, 'paper.pdf'));
assert.equal(pdf.subarray(0, 5).toString(), '%PDF-');
const digest = createHash('sha256').update(pdf).digest('hex');
const fileName = 'counterfactual-tool-ranking.pdf';
const metadata = {
  title,
  authors: [{ name: 'Jiapeng Li', affiliation: 'Microsoft' }],
  abstract,
  primary_category_proposed: 'cs.LG',
  comments: 'Working paper, version 2. Includes synthetic execution studies, BFCL-derived function-selection experiments, and local open-weight LLM baselines. Code and artifacts: https://github.com/jaxblack/counterfactual-tool-ranking',
  journal_reference: null,
  doi: null,
  license: 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/',
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
  license_status: 'selected_by_author_not_yet_granted_to_arxiv',
  submission_agreement_accepted: false,
  pending: [
    'Author creates and verifies their own arXiv account.',
    'Complete subject-area endorsement if requested by arXiv.',
    'Author reviews and accepts the arXiv submission agreement on the website.',
    'Review the processed PDF and metadata, then explicitly submit through the author account.',
  ],
};
await fs.mkdir(destination, { recursive: true });
await fs.writeFile(path.join(destination, fileName), pdf);
await fs.writeFile(path.join(destination, 'abstract.txt'), abstract + '\n');
await fs.writeFile(path.join(destination, 'metadata.json'), JSON.stringify(metadata, null, 2) + '\n');
await fs.writeFile(path.join(destination, 'status.json'), JSON.stringify(status, null, 2) + '\n');
console.log(JSON.stringify({ status: status.status, pdf: fileName, bytes: pdf.length, sha256: digest, abstract_characters: abstract.length }, null, 2));