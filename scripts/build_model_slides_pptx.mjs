import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const runtimeNodeModules = process.env.RUNTIME_NODE_MODULES
  ?? "/Users/proskurin-dmi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const { Presentation, PresentationFile } = await import(
  pathToFileURL(path.join(runtimeNodeModules, "@oai/artifact-tool/dist/artifact_tool.mjs")).href,
);

const workspaceDir = "/Users/proskurin-dmi/Documents/ai product hack/AlphaTransfer";
const skillDir = "/Users/proskurin-dmi/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const runtimePython = "/Users/proskurin-dmi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const buildDir = path.join(workspaceDir, ".codex-pptx-build");
const outputDir = path.join(workspaceDir, "product_artifacts");
const finalPath = path.join(outputDir, "AlphaTransfer_model_slides_final.pptx");
const candidatePath = path.join(buildDir, "candidate.pptx");

const { finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const W = 1600;
const H = 900;
const C = {
  bg: "#F3F4F5",
  white: "#FFFFFF",
  ink: "#171717",
  navy: "#0B1F35",
  gray: "#7C7F88",
  gray2: "#A8ABB2",
  line: "#DFE2E6",
  soft: "#ECEEF1",
  red: "#EF3124",
  redSoft: "#FBE9E7",
  green: "#1F8A55",
  greenSoft: "#E7F3ED",
  blue: "#3974C9",
  blueSoft: "#E9F0FA",
  amber: "#C67A12",
  amberSoft: "#FAF0DF",
  purple: "#7652B8",
  purpleSoft: "#EEE9F8",
};
const FONT = "Inter";

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addShape(slide, geometry, x, y, w, h, fill, opts = {}) {
  return slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h, rotation: opts.rotation ?? 0 },
    fill: fill ?? "none",
    line: opts.line ?? { fill: "none", width: 0 },
    borderRadius: opts.radius,
    shadow: opts.shadow,
    name: opts.name,
  });
}

function addText(slide, text, x, y, w, h, size, color = C.ink, opts = {}) {
  const shape = addShape(slide, "textbox", x, y, w, h, "none", { name: opts.name });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: size,
    bold: opts.bold ?? false,
    color,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
    wrap: "square",
    insets: opts.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
    lineSpacing: opts.lineSpacing,
  };
  return shape;
}

function addPill(slide, text, x, y, w, fill, color) {
  const pill = addShape(slide, "roundRect", x, y, w, 28, fill, { radius: 14 });
  pill.text = text;
  pill.text.style = {
    typeface: FONT,
    fontSize: 11,
    bold: true,
    color,
    alignment: "center",
    verticalAlignment: "middle",
    autoFit: "shrinkText",
    insets: { top: 1, right: 8, bottom: 1, left: 8 },
  };
  return pill;
}

function addHeader(slide, eyebrow, title, subtitle) {
  slide.background.fill = C.bg;
  addShape(slide, "rect", 0, 0, W, 5, C.red);
  addShape(slide, "ellipse", 1558, 0, 22, 22, C.red);
  addText(slide, eyebrow, 68, 38, 500, 22, 12, C.red, { bold: true });
  addText(slide, title, 68, 67, 1464, 58, 42, C.ink, { bold: true });
  addText(slide, subtitle, 68, 126, 1464, 34, 16, C.gray);
}

function addFooter(slide, page) {
  addShape(slide, "line", 68, 846, 1464, 0, "none", {
    line: { style: "solid", fill: C.red, width: 1 },
  });
  addShape(slide, "ellipse", 1290, 841, 10, 10, C.red);
  addText(slide, "AlphaTransfer", 68, 860, 220, 20, 11, C.red, { bold: true });
  addText(slide, `${page} / 2`, 1450, 860, 82, 20, 11, C.gray, { align: "right" });
}

function addMetric(slide, x, y, value, label, color = C.ink, w = 150) {
  addText(slide, value, x, y, w, 47, 32, color, { bold: true });
  addText(slide, label, x, y + 43, w, 27, 11, C.gray, { lineSpacing: 1 });
}

function addYearCell(slide, x, y, year, lift, delta, w) {
  addShape(slide, "roundRect", x, y, w, 48, C.soft, { radius: 10 });
  addText(slide, year, x + 12, y + 8, 45, 16, 10, C.gray, { bold: true });
  addText(slide, lift, x + 62, y + 7, 72, 18, 12, C.red, { bold: true });
  addText(slide, delta, x + 62, y + 26, 85, 14, 9, C.gray);
}

function addModelPanel(slide, cfg) {
  addShape(slide, "roundRect", cfg.x, 184, 714, 326, C.white, { radius: 18 });
  addPill(slide, cfg.chip, cfg.x + 24, 205, 106, C.greenSoft, C.green);
  addText(slide, cfg.title, cfg.x + 24, 243, 640, 36, 23, C.navy, { bold: true });
  addText(slide, cfg.desc, cfg.x + 24, 280, 650, 30, 13, C.gray);

  addMetric(slide, cfg.x + 24, 326, cfg.lift, "lift к случайному дню", C.red, 175);
  addMetric(slide, cfg.x + 250, 326, cfg.hit, `hit rate · base ${cfg.base}`, C.ink, 165);
  addMetric(slide, cfg.x + 495, 326, cfg.delta, "выгода момента", C.green, 175);

  const rowY = 407;
  addMetric(slide, cfg.x + 24, rowY, cfg.coverage, "недель с 1–2 сигналами", C.ink, 165);
  addMetric(slide, cfg.x + 225, rowY, cfg.freq, "сигнала в неделю", C.ink, 145);
  addMetric(slide, cfg.x + 410, rowY, cfg.auc, "ROC AUC", C.ink, 105);
  addMetric(slide, cfg.x + 560, rowY, cfg.brier, "Brier", C.ink, 105);

  addText(slide, "УСТОЙЧИВОСТЬ ПО ГОДАМ · LIFT / ВЫГОДА", cfg.x + 24, 466, 370, 18, 10, C.gray, { bold: true });
  const cellW = 202;
  addYearCell(slide, cfg.x + 24, 488, "2024", cfg.years[0][0], cfg.years[0][1], cellW);
  addYearCell(slide, cfg.x + 256, 488, "2025", cfg.years[1][0], cfg.years[1][1], cellW);
  addYearCell(slide, cfg.x + 488, 488, "2026", cfg.years[2][0], cfg.years[2][1], cellW);
}

// Slide 1: model benchmarks.
{
  const slide = presentation.slides.add();
  addHeader(
    slide,
    "МОДЕЛЬНЫЕ БЕНЧМАРКИ",
    "Два горизонта, два режима сигнала",
    "Rolling OOT 2024–2026 · KZT · ежегодное переобучение только на доступной истории",
  );

  addModelPanel(slide, {
    x: 68,
    chip: "H3 · 3 СЕССИИ",
    title: "Регулярный сигнал",
    desc: "Для пользователей, которые переводят реже и могут выбирать момент",
    lift: "1,504×", hit: "49,0%", base: "32,9%", delta: "+54,2 б.п.",
    coverage: "85,9%", freq: "1,12", auc: "0,765", brier: "0,178",
    years: [["1,609×", "+57,5 б.п."], ["1,452×", "+40,3 б.п."], ["1,441×", "+73,8 б.п."]],
  });
  addModelPanel(slide, {
    x: 818,
    chip: "H5 · 5 СЕССИЙ",
    title: "Редкий, более избирательный сигнал",
    desc: "Для пользователей, которым нужно меньше уведомлений и более длинное окно",
    lift: "1,986×", hit: "52,8%", base: "26,5%", delta: "+95,6 б.п.",
    coverage: "56,3%", freq: "0,66", auc: "0,737", brier: "0,170",
    years: [["2,063×", "+96,6 б.п."], ["2,241×", "+74,8 б.п."], ["1,711×", "+125,7 б.п."]],
  });

  addShape(slide, "roundRect", 68, 553, 1464, 224, C.white, { radius: 18 });
  addPill(slide, "CLOSING H3", 92, 577, 118, C.greenSoft, C.green);
  addText(slide, "Уточняет сообщение: «окно может закрыться»", 92, 616, 525, 34, 22, C.navy, { bold: true });
  addText(
    slide,
    "Дополнительный слой усиливает часть сигналов H3 и не создаёт новых уведомлений",
    92, 654, 525, 42, 13, C.gray,
  );
  addMetric(slide, 665, 606, "61 из 151", "усиленный H3-сигнал", C.ink, 170);
  addMetric(slide, 870, 606, "67,2%", "hit rate · base 49,8%", C.ink, 175);
  addMetric(slide, 1080, 606, "1,351×", "lift", C.red, 135);
  addMetric(slide, 1260, 606, "+87,1 б.п.", "изменение к концу окна", C.green, 200);
  addShape(slide, "line", 665, 696, 770, 0, "none", { line: { style: "solid", fill: C.line, width: 1 } });
  addText(slide, "Brier 0,227 · ROC AUC 0,676 · дополнительных контактов 0", 665, 712, 720, 24, 12, C.gray);
  addText(slide, "Показатели относятся к rolling OOT. Финальный refit ещё не имеет будущего confirmatory-теста.", 92, 730, 525, 30, 11, C.gray);

  addFooter(slide, 1);
  slide.speakerNotes.textFrame.setText(
    "Источники чисел: docs/assets/model-metrics/metrics_snapshot.csv и README.md. Показатели рассчитаны на rolling OOT 2024–2026. H3 и H5 имеют разные целевые события и разные base hit rate.",
  );
}

function addSourceCard(slide, cfg) {
  addShape(slide, "roundRect", cfg.x, 198, cfg.w, 304, C.white, { radius: 18 });
  addPill(slide, cfg.source, cfg.x + 18, 218, cfg.pillW, cfg.soft, cfg.color);
  addText(slide, cfg.title, cfg.x + 18, 260, cfg.w - 36, 58, 19, C.navy, { bold: true });
  addText(slide, cfg.body, cfg.x + 18, 328, cfg.w - 36, 88, 13, C.gray, { lineSpacing: 1.05 });
  addShape(slide, "line", cfg.x + 18, 436, cfg.w - 36, 0, "none", {
    line: { style: "solid", fill: C.line, width: 1 },
  });
  addText(slide, cfg.features, cfg.x + 18, 454, cfg.w - 36, 36, 10, cfg.color, { bold: true });
}

function addFlowBox(slide, x, y, w, n, title, body, accent = false) {
  const box = addShape(slide, "roundRect", x, y, w, 120, accent ? C.red : C.white, { radius: 16 });
  addPill(slide, `ШАГ ${String(n).padStart(2, "0")}`, x + 16, y + 14, 72, accent ? C.white : C.greenSoft, accent ? C.red : C.green);
  addText(slide, title, x + 16, y + 51, w - 32, 28, 18, accent ? C.white : C.navy, { bold: true });
  addText(slide, body, x + 16, y + 82, w - 32, 28, 11, accent ? C.white : C.gray);
  return box;
}

// Slide 2: exact triggers and delivery path.
{
  const slide = presentation.slides.add();
  addHeader(
    slide,
    "СИГНАЛЬНЫЙ СЛОЙ",
    "Триггеры, которые учитывает модель",
    "Пять источников дают 33 измеримых признака. Каждый признак известен к моменту принятия решения.",
  );

  const gap = 18;
  const cardW = 278;
  const start = 68;
  const cards = [
    {
      source: "ЦБ РФ", pillW: 66, color: C.blue, soft: C.blueSoft,
      title: "RUB/KZT у локального минимума",
      body: "Модель оценивает положение курса за месяц, квартал, полгода и год, а также изменения за 1–60 сессий.",
      features: "ret 1/3/5/10/20/60 · percentile 20/60/120/252",
    },
    {
      source: "MOEX", pillW: 68, color: C.green, soft: C.greenSoft,
      title: "CNY/RUB меняется после фиксинга",
      body: "Цена закрытия CNY/RUB сравнивается с фиксингом той же сессии. Расхождение отражает более свежее движение рубля.",
      features: "CNY/RUB close − fixing",
    },
    {
      source: "OXR", pillW: 58, color: C.amber, soft: C.amberSoft,
      title: "Внешний RUB/KZT расходится с ЦБ",
      body: "Внешний кросс-курс сравнивается с официальным. Модель учитывает величину, изменение, аномальность и свежесть расхождения.",
      features: "basis · Δ1 · Δ5 · z-score 20 · age · available",
    },
    {
      source: "HALYK", pillW: 72, color: C.red, soft: C.redSoft,
      title: "Halyk переоценивает RUB/KZT",
      body: "Банковская цена сопоставляется с курсом ЦБ. Отдельно учитываются разница условий и изменения RUB и USD за 1 и 5 сессий.",
      features: "basis · personal/legal gap · RUB/USD ret 1/5",
    },
    {
      source: "TREASURY", pillW: 96, color: C.purple, soft: C.purpleSoft,
      title: "Меняются ожидания по инфляции в США",
      body: "Рыночные инфляционные ожидания задают глобальный режим доллара и помогают различать одинаковые локальные движения.",
      features: "T10YIE · T5YIFR · изменения",
    },
  ];
  cards.forEach((card, i) => addSourceCard(slide, { ...card, x: start + i * (cardW + gap), w: cardW }));

  addShape(slide, "line", 68, 527, 1464, 0, "none", { line: { style: "solid", fill: C.line, width: 1 } });
  addText(
    slide,
    "Задержка публикации, возраст и пропуски входят в данные. Модель не получает будущие значения.",
    68, 539, 1464, 24, 12, C.green, { bold: true, align: "center" },
  );

  const y = 590;
  const boxes = [
    addFlowBox(slide, 68, y, 252, 1, "Пять источников", "ЦБ, MOEX, OXR, Halyk, Treasury"),
    addFlowBox(slide, 364, y, 214, 2, "33 признака", "уровни, изменения, расхождения"),
    addFlowBox(slide, 622, y, 214, 3, "TabM H3 / H5", "вероятность события через 3 или 5 сессий"),
    addFlowBox(slide, 880, y, 280, 4, "Порог и частота", "сравнение с 63 прошлыми оценками"),
    addFlowBox(slide, 1204, y, 328, 5, "Пуш с фактом", "или тишина, если прогноз недостаточно силён", true),
  ];
  for (let i = 0; i < boxes.length - 1; i++) {
    slide.shapes.connect(boxes[i], boxes[i + 1], {
      kind: "straight",
      fromSide: "right",
      toSide: "left",
      line: { style: "solid", fill: C.red, width: 2 },
      tail: { type: "triangle", width: "sm", length: "sm" },
    });
  }
  addShape(slide, "roundRect", 68, 730, 1464, 74, C.white, { radius: 14 });
  addText(slide, "Что увидит клиент", 90, 748, 200, 22, 13, C.red, { bold: true });
  addText(
    slide,
    "Проверяемый факт о положении курса в историческом диапазоне. Для части H3-сигналов CLOSING добавляет: «окно может закрыться».",
    285, 743, 1215, 34, 16, C.navy, { bold: true, valign: "middle" },
  );

  addFooter(slide, 2);
  slide.speakerNotes.textFrame.setText(
    "Состав признаков и ограничения доступности описаны в docs/MODEL_METRICS.md. Клиентская формулировка строится из проверяемых наблюдений, а не из предположений о внутренней причине прогноза TabM.",
  );
}

await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 2,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
};
const result = await finalizePresentation({
  ...requirements,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "15240000,8572500",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "AlphaTransfer_model_slides_final.validation.json"),
});

console.log(JSON.stringify({ finalPath, result }, null, 2));
