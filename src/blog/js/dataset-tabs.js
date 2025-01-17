// Configuration class for table systems
class TableConfig {
  constructor({
    id,
    title,
    datasets,
    metrics,
    pathTemplate,
    containerSelector,
  }) {
    this.id = id;
    this.title = title;
    this.datasets = datasets;
    this.metrics = metrics;
    this.pathTemplate = pathTemplate;
    this.containerSelector = containerSelector;
  }
}

// Function to create a table system
function createTableSystem(config) {
  const container = document.querySelector(config.containerSelector);
  if (!container) return;

  // Create the basic structure
  container.innerHTML = `
    <hr />
    <div class="l-screen-inset" style="margin-left: 5em" id="tab-system-${config.id}">
      <b style="margin-bottom: 1em">${config.title}</b>
      <div class="dataset-tabs" style="margin-bottom: 1em" id="tabs-${config.id}"></div>
      <div class="dataset-content" id="content-${config.id}"></div>
    </div>
    <hr />
  `;

  generateTabs(config);
  generateContent(config);
}

// Function to generate tabs
function generateTabs(config) {
  const tabsContainer = document.getElementById(`tabs-${config.id}`);

  config.datasets.forEach((dataset, index) => {
    const button = document.createElement("button");
    button.className = `tab-button${index === 0 ? " active important" : ""}`;
    button.onclick = () => showDataset(dataset.id, config.id);

    button.innerHTML = `
      <i class="fas ${dataset.icon}" style="color: #666; width: 1.5em; text-align: center"></i>
      ${dataset.name}
    `;

    tabsContainer.appendChild(button);
  });
}

// Function to generate content
function generateContent(config) {
  const contentContainer = document.getElementById(`content-${config.id}`);
  const isSingleMetric = config.metrics.length === 1;

  config.datasets.forEach((dataset, index) => {
    const div = document.createElement("div");
    div.className = `tab-content${index === 0 ? " active" : ""}`;
    div.id = `${dataset.id}-content-${config.id}`;


    div.innerHTML = `
      <div class="dataset-grid" style="
        display: ${isSingleMetric ? 'block' : 'grid'};
        text-align: center;
        margin: 0 auto;
      ">
        ${config.metrics
          .map(
            (metric) => `
          <figure style="
            width: 100%;
            margin: 0 auto;
          ">
            <b>${metric.name}</b>
            <iframe
              style="
                width: 100%;
                max-width: ${isSingleMetric ? '800px' : '100%'};
                margin: 0 auto;
                width: 900px;
                height: 700px;
                margin: 0 auto;
                display: block;
                transform: scale(1.0);  // Forces a specific scale
                sandbox="allow-scripts allow-same-origin"  // Add specific permissions and sus but works
                loading="lazy"
                display: block;
              "
              src="${generatePath(config.pathTemplate, {
                dataset: dataset.id,
                metric: metric.id,
              })}"
              scrolling="no"></iframe>
            <figcaption>
              ${metric.name} loss visualization across model pairs
              ${metric.isTrainingObjective ? "(Training Objective)" : ""} - ${dataset.name} Dataset
            </figcaption>
          </figure>
        `,
          )
          .join("")}
      </div>
    `;

    contentContainer.appendChild(div);
  });
}

// Function to show selected dataset
function showDataset(datasetId, systemId) {
  // Hide all content
  document
    .querySelectorAll(`#tab-system-${systemId} .tab-content`)
    .forEach((content) => {
      content.classList.remove("active");
    });

  // Show selected content
  document
    .getElementById(`${datasetId}-content-${systemId}`)
    .classList.add("active");

  // Update tab styling
  document
    .querySelectorAll(`#tab-system-${systemId} .tab-button`)
    .forEach((button) => {
      button.classList.remove("active");
    });
  document
    .querySelector(`#tab-system-${systemId} [onclick*="${datasetId}"]`)
    .classList.add("active");
}

// Helper function to generate paths using template
function generatePath(template, params) {
  let path = template;
  Object.entries(params).forEach(([key, value]) => {
    path = path.replace(`{${key}}`, value);
  });
  return path;
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  // Example configuration for your first table system
  const mseConfig = new TableConfig({
    id: 1,
    title: "Validation MSE and MAE of Affine Stitch Across Datasets",
    datasets: [
      { id: "weightedmean", name: "Weighted Average", icon: "fa-calculator" },
      { id: "arguana", name: "ArguAna", icon: "fa-comments" },
      { id: "fiqa", name: "FiQA", icon: "fa-chart-line" },
      { id: "scidocs", name: "SciDocs", icon: "fa-flask" },
      { id: "nfcorpus", name: "NFCorpus", icon: "fa-heartbeat" },
      { id: "hotpotqa", name: "HotpotQA", icon: "fa-question-circle" },
      { id: "trec-covid", name: "TREC-COVID", icon: "fa-virus" },
    ],
    metrics: [
      {
        id: "mse",
        name: "Mean Squared Error (MSE)",
        isTrainingObjective: true,
      },
      {
        id: "mae",
        name: "Mean Absolute Error (MAE)",
        isTrainingObjective: false,
      },
    ],
    pathTemplate:
      "./figs/adriano/figs/html/{dataset}_olsaffine_{metric}_withlog_validation.html",
    containerSelector: "#table-system-1",
  });

  // Example configuration for your second table system
  const r2Config = new TableConfig({
    id: 2,
    title: "R2 MSE and Percent of MAE Explained",
    datasets: mseConfig.datasets, // Reuse the same datasets
    metrics: [
      { id: "r2_mse", name: "R2 MSE", isTrainingObjective: true },
      {
        id: "r2_mae",
        name: "Percent of MAE Explained",
        isTrainingObjective: false,
      },
    ],
    pathTemplate:
      "./figs/adriano/figs/html/{dataset}_olsaffine_{metric}_withoutlog_validation.html",
    containerSelector: "#table-system-2",
  });

// Example configuration for your third table system
const mseMultiLayer = new TableConfig({
    id: 3,
    title: "MSE Validation on Mutli-Layered NNs",
    datasets: mseConfig.datasets,  // Reuse the same datasets
    metrics: [
        { 
            id: "mse",
            name: "Mean Squared Error (MSE)",
            isTrainingObjective: true 
        }
    ],
    pathTemplate: "./figs/adriano/figs/html/{dataset}_lord_farquad_was_heren copy.html",
    containerSelector: "#table-system-3"
    });

    
    // Create both table systems
    createTableSystem(mseConfig);
    createTableSystem(r2Config);
    createTableSystem(mseMultiLayer);
});
