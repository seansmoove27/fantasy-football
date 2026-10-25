// Paste into the console (or run via Claude in Chrome) on a FantasyPros weekly rankings page,
// e.g. https://www.fantasypros.com/nfl/rankings/k.php?week=7 or dst.php?week=7.
// Selects "<last season> In-Season Accuracy: Top 10 <pos>" and returns "pos|week|TEAM:rank,..." for
// data/espn/fp_top10_<season>.csv (the previous season's top 10, so the expert set is chosen out of sample).
(async () => {
  const pos = location.pathname.includes('dst') ? 'DST' : 'K';
  for (let i = 0; i < 20 && !document.querySelector('li.select-advanced__item'); i++) await new Promise(r => setTimeout(r, 500));
  const lis = [...document.querySelectorAll('li.select-advanced__item')];
  const h = lis.findIndex(li => /20\d\d In-Season Accuracy/.test(li.textContent) && !li.textContent.includes(String(new Date().getFullYear())));
  const target = lis.slice(h).find(li => new RegExp('Top 10 ' + pos + '$').test(li.textContent.trim()));
  (target.querySelector('button') || target).click();
  await new Promise(r => setTimeout(r, 3500));
  const rows = [...document.querySelectorAll('table tbody tr')]
    .map(r => r.innerText.replace(/\s+/g, ' ').match(/^(\d+) .*?\(([A-Z]{2,3})\)/))
    .filter(Boolean).map(m => m[2] + ':' + m[1]);
  return pos + '|' + new URLSearchParams(location.search).get('week') + '|' + rows.join(',');
})();
