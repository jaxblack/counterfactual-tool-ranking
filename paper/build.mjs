import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import MarkdownIt from 'markdown-it';
import texmath from 'markdown-it-texmath';
import katex from 'katex';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(directory);
const aggregate = JSON.parse(await fs.readFile(path.join(root, 'artifacts/v1/aggregate.json'), 'utf8'));
const mcpSummary = JSON.parse(await fs.readFile(path.join(root, 'artifacts/mcp-v1/summary.json'), 'utf8'));
const extension = JSON.parse(await fs.readFile(path.join(root, 'artifacts/v2/aggregate.json'), 'utf8'));
const diagnostics = JSON.parse(await fs.readFile(path.join(root, 'artifacts/v2/diagnostics.json'), 'utf8'));
const publicSummaries = await Promise.all([7, 17, 23, 31, 47].map(async seed =>
  JSON.parse(await fs.readFile(path.join(root, `artifacts/v2/public-native-${seed}/summary.json`), 'utf8'))));
const original = await fs.readFile(path.join(directory, 'manuscript.md'), 'utf8');
assert.equal(aggregate.run_count, 45);
assert.equal(extension.run_count, 45);
assert.equal(aggregate.manifest.seeds.length, 5);
const settingNames = Object.keys(aggregate.settings);
const learnedPolicies = ['rules', 'direct', 'ips', 'dr', 'conservative_direct', 'conservative_dr'];
const format = (value, digits = 4) => Number(value).toFixed(digits);
const table = (headers, rows) => [
  `| ${headers.join(' | ')} |`,
  `| ${headers.map(() => '---').join(' | ')} |`,
  ...rows.map(row => `| ${row.join(' | ')} |`),
].join('\n');
const utilityTable = table(
  ['Setting', 'Rules', 'Direct', 'IPS', 'DR', 'Cons. direct', 'Cons. DR'],
  settingNames.map(name => [name.replaceAll('_', ' '), ...learnedPolicies.map(policy => {
    const value = aggregate.settings[name].policies[policy].utility;
    return `${format(value.mean)} (${format(value.std)})`;
  })]),
);
const opeTable = table(
  ['Setting', 'Direct', 'IPS', 'SNIPS', 'DR', 'Cases'],
  settingNames.map(name => {
    const values = aggregate.settings[name].ope_absolute_errors;
    return [name.replaceAll('_', ' '), ...['direct', 'ips', 'snips', 'dr'].map(method => format(values[method].mean)), values.dr.count];
  }),
);
const noisyTable = table(
  ['Policy', 'Coverage %', 'Success/all %', 'Unsafe/all %', 'Cost/success', 'Utility'],
  ['schema_match', ...learnedPolicies, 'oracle'].map(name => {
    const metrics = aggregate.settings.noisy.policies[name];
    return [name.replaceAll('_', ' '), ...['coverage', 'success_all_tasks', 'unsafe_rate'].map(metric => format(metrics[metric].mean * 100, 2)),
      format(metrics.cost_per_success.mean), format(metrics.utility.mean)];
  }),
);
const number = (setting, policy, metric = 'utility') => format(aggregate.settings[setting].policies[policy][metric].mean);
const controlSettings = ['clean', 'noisy', 'shifted', 'linear', 'cost_sensitive', 'latency_sensitive'];
const controlsTable = table(
  ['Setting', 'Nominal DM', 'Full DM', 'Component DM', 'Nominal DR', 'Full DR'],
  controlSettings.map(setting => {
    const errors = extension.synthetic[setting].ope_errors;
    return [setting.replaceAll('_', ' '), ...[
      ['nominal_direct', 'direct'], ['full_direct', 'direct'], ['component_direct', 'direct'],
      ['nominal_direct', 'dr'], ['full_direct', 'dr'],
    ].map(([nuisance, estimator]) => format(errors[nuisance][estimator].mean))];
  }),
);
const publicTable = table(
  ['Policy', 'Selection accuracy %', 'Balanced accuracy %', 'Coverage %'],
  Object.entries(diagnostics.native_means).map(([policy, values]) => [
    policy.replaceAll('_', ' '), ...['accuracy', 'balanced_accuracy', 'coverage'].map(metric => format(values[metric] * 100, 2)),
  ]),
);
const multipleCandidateTable = table(
  ['Policy', 'Selection accuracy %', 'Balanced accuracy %', 'Coverage %'],
  Object.entries(diagnostics.multi_candidate_means).map(([policy, values]) => [
    policy.replaceAll('_', ' '), ...['accuracy', 'balanced_accuracy', 'coverage'].map(metric => format(values[metric] * 100, 2)),
  ]),
);
const modelTable = table(
  ['Model', 'Tasks', 'Should-call correct', 'Should-abstain correct', 'JSON valid %', 'Schema valid/known call %'],
  Object.entries(diagnostics.models).map(([name, model]) => [
    name, model.tasks, `${model.per_category.multiple.correct}/${model.per_category.multiple.tasks}`,
    `${model.per_category.irrelevance.correct}/${model.per_category.irrelevance.tasks}`,
    format(model.format_validity * 100, 2), format(model.schema_validity_among_known_calls * 100, 2),
  ]),
);
const matchedTable = table(
  ['Matched model subset', 'Policy', 'Selection accuracy %', 'Balanced accuracy %'],
  Object.entries(diagnostics.matched_means).flatMap(([model, policies]) =>
    Object.entries(policies).map(([policy, values]) => [model, policy.replaceAll('_', ' '), format(values.accuracy * 100, 2), format(values.balanced_accuracy * 100, 2)])),
);
const contrastTable = table(
  ['Comparison against TF-IDF', 'Both absolute values identified', 'Gain point-identified', 'Mean identification width'],
  ['dr', 'blanket_support_abstain', 'disagreement_fallback'].map(policy => {
    const value = extension.public.support_gap.contrasts[policy];
    return [policy.replaceAll('_', ' '), `${value.absolute_identified_runs}/5`, `${value.point_identified_runs}/5`, format(value.mean_identification_width)];
  }),
);
const substitutions = {
  utility_table: utilityTable,
  ope_table: opeTable,
  noisy_table: noisyTable,
  decisions: aggregate.selected_decision_count.toLocaleString('en-US'),
  task_conditions: aggregate.test_task_condition_count.toLocaleString('en-US'),
  noisy_direct: number('noisy', 'direct'),
  noisy_dr: number('noisy', 'dr'),
  noisy_rules: number('noisy', 'rules'),
  noisy_conservative: number('noisy', 'conservative_dr'),
  small_direct: number('small_data', 'direct'),
  small_dr: number('small_data', 'dr'),
  linear_direct: number('linear', 'direct'),
  linear_dr: number('linear', 'dr'),
  linear_ips: number('linear', 'ips'),
  shifted_dm_error: format(aggregate.settings.shifted.ope_absolute_errors.direct.mean),
  shifted_dr_error: format(aggregate.settings.shifted.ope_absolute_errors.dr.mean),
  linear_dm_error: format(aggregate.settings.linear.ope_absolute_errors.direct.mean),
  linear_dr_error: format(aggregate.settings.linear.ope_absolute_errors.dr.mean),
  small_dm_error: format(aggregate.settings.small_data.ope_absolute_errors.direct.mean),
  small_dr_error: format(aggregate.settings.small_data.ope_absolute_errors.dr.mean),
  linear_paired: format(aggregate.settings.linear.paired_seed_differences['dr-minus-direct'].mean),
  mcp_calls: String(mcpSummary.execution.calls),
  mcp_p50: format(mcpSummary.execution.roundtrip_p50_ms, 3),
  mcp_p95: format(mcpSummary.execution.roundtrip_p95_ms, 3),
  mcp_protocol: mcpSummary.execution.protocol,
  v2_controls_table: controlsTable,
  v2_public_table: publicTable,
  v2_multi_candidate_table: multipleCandidateTable,
  v2_models_table: modelTable,
  v2_matched_table: matchedTable,
  v2_contrast_table: contrastTable,
  v2_linear_full_dm: format(extension.synthetic.linear.ope_errors.full_direct.direct.mean),
  v2_linear_full_dr: format(extension.synthetic.linear.ope_errors.full_direct.dr.mean),
  v2_shift_full_dm: format(extension.synthetic.shifted.ope_errors.full_direct.direct.mean),
  v2_shift_full_dr: format(extension.synthetic.shifted.ope_errors.full_direct.dr.mean),
  v2_public_direct_balanced: format(diagnostics.native_means.direct.balanced_accuracy * 100, 2),
  v2_public_dr_balanced: format(diagnostics.native_means.dr.balanced_accuracy * 100, 2),
  v2_public_tfidf_balanced: format(diagnostics.native_means.tfidf.balanced_accuracy * 100, 2),
  v2_multi_direct_balanced: format(diagnostics.multi_candidate_means.direct.balanced_accuracy * 100, 2),
  v2_multi_tfidf_balanced: format(diagnostics.multi_candidate_means.tfidf.balanced_accuracy * 100, 2),
  v2_min_test_tasks: String(Math.min(...publicSummaries.map(summary => summary.split.sizes.test))),
  v2_max_test_tasks: String(Math.max(...publicSummaries.map(summary => summary.split.sizes.test))),
  v2_min_test_groups: String(Math.min(...publicSummaries.map(summary => summary.split.group_counts.test))),
  v2_max_test_groups: String(Math.max(...publicSummaries.map(summary => summary.split.group_counts.test))),
  v2_llm_act_count: String(diagnostics.models['qwen-0.5b'].per_category.multiple.tasks),
  v2_llm_abstain_count: String(diagnostics.models['qwen-0.5b'].per_category.irrelevance.tasks),
};
const markdown = original.replace(/\{\{([a-z0-9_]+)\}\}/g, (_, name) => {
  assert.ok(name in substitutions, `Unknown generated result: ${name}`);
  return substitutions[name];
});
assert.ok(!markdown.includes('{{'), 'Unresolved result placeholder');
await fs.writeFile(path.join(directory, 'paper.md'), markdown);
const renderer = new MarkdownIt({ html: false, linkify: true }).use(texmath, {
  engine: katex, delimiters: 'dollars', katexOptions: { throwOnError: true, strict: 'error', trust: false },
});
let content = renderer.render(markdown);
for (const match of [...content.matchAll(/src="(\.\.\/artifacts\/[^"<>]+\.png)"/g)]) {
  const bytes = await fs.readFile(path.resolve(directory, match[1]));
  content = content.replaceAll(match[0], `src="data:image/png;base64,${bytes.toString('base64')}"`);
}
content = content.replaceAll('<table>', '<div class="table-wrap"><table>').replaceAll('</table>', '</table></div>');
let mathCss = await fs.readFile(path.join(directory, 'node_modules/katex/dist/katex.min.css'), 'utf8');
for (const match of [...mathCss.matchAll(/url\((fonts\/[^)]+)\)/g)]) {
  const font = await fs.readFile(path.join(directory, 'node_modules/katex/dist', match[1]));
  const extension = path.extname(match[1]).slice(1);
  mathCss = mathCss.replaceAll(match[0], `url(data:font/${extension};base64,${font.toString('base64')})`);
}
const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Counterfactual Tool Ranking: Working Paper</title><style>${mathCss}
*{box-sizing:border-box;letter-spacing:0}body{margin:0;color:#1c2225;background:#f2f4f5;font:16px/1.6 Charter,"Palatino Linotype",serif}
main{max-width:940px;margin:auto;padding:42px 46px 64px;background:white}
h1,h2,h3{font-family:"Avenir Next","Trebuchet MS",sans-serif;line-height:1.25;break-after:avoid}
h1{font-size:28px;margin:0 0 18px}h2{font-size:21px;margin:34px 0 12px}h3{font-size:17px;margin:24px 0 10px}
p{margin:12px 0}a{color:#155675;overflow-wrap:anywhere}img{display:block;max-width:100%;height:auto;margin:20px auto}
table{border-collapse:collapse;width:100%;font-size:11px;line-height:1.4;font-family:"Avenir Next","Trebuchet MS",sans-serif}
th,td{padding:7px 6px;border-bottom:1px solid #d9dfe2;text-align:left;vertical-align:top}th{background:#edf3f4}
.table-wrap{overflow-x:auto;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;background:#f4f6f7;font-size:11px}
code{font-family:Menlo,Consolas,monospace;font-size:.84em;overflow-wrap:anywhere}blockquote{margin:16px 0;padding-left:16px;border-left:3px solid #568594}
.katex-display{overflow-x:auto;overflow-y:hidden;padding:6px 0}.katex{font-size:1.04em}li{margin:4px 0}
@media(max-width:600px){main{padding:24px 18px}h1{font-size:24px}h2{font-size:20px}}
@page{size:A4;margin:18mm 16mm 18mm}
@media print{body{background:white;font-size:10pt;line-height:1.45}main{max-width:none;padding:0}h1{font-size:21pt}h2{font-size:14pt;margin-top:22px}h3{font-size:11pt}table{font-size:7pt}img{max-height:190mm;object-fit:contain;break-inside:avoid}.table-wrap{overflow:visible}tr{break-inside:avoid}.katex-display{overflow:visible;font-size:9pt}p{orphans:3;widows:3}}
</style></head><body><main>${content}</main></body></html>`;
const htmlPath = path.join(directory, 'paper.html');
await fs.writeFile(htmlPath, html);
await fs.writeFile(path.join(directory, 'claims.json'), JSON.stringify(substitutions, null, 2) + '\n');
if (process.argv.includes('--pdf')) {
  const { chromium } = await import('playwright');
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 850 } });
    await page.route(/^https?:/, route => route.abort());
    await page.goto(pathToFileURL(htmlPath).href);
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator('.katex-error').count(), 0, 'Math rendering error');
    assert.ok(await page.locator('.katex').count() >= 15, 'Expected rendered math expressions');
    assert.ok(await page.locator('img').evaluateAll(images => images.every(image => image.complete && image.naturalWidth > 0)), 'Broken figure');
    await fs.mkdir(path.join(directory, '.preview'), { recursive: true });
    await page.screenshot({ path: path.join(directory, '.preview/desktop.png') });
    await page.pdf({ path: path.join(directory, 'paper.pdf'), format: 'A4', printBackground: true, preferCSSPageSize: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), 'Mobile document overflow');
    await page.screenshot({ path: path.join(directory, '.preview/mobile.png') });
  } finally {
    await browser.close();
  }
}
console.log('Built paper.md, offline paper.html, claims.json' + (process.argv.includes('--pdf') ? ', and paper.pdf' : ''));