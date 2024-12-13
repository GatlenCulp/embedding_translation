// Define the dataset configurations
const datasets = [
  {
    id: 'average',
    name: 'Weighted Average',
    icon: 'fa-calculator',
    isDefault: true
  },
  {
    id: 'arguana',
    name: 'ArguAna',
    icon: 'fa-comments'
  },
  {
    id: 'fiqa',
    name: 'FiQA', 
    icon: 'fa-chart-line'
  },
  {
    id: 'scidocs',
    name: 'SciDocs',
    icon: 'fa-flask'
  },
  {
    id: 'nfcorpus',
    name: 'NFCorpus',
    icon: 'fa-heartbeat'
  },
  {
    id: 'hotpotqa',
    name: 'HotpotQA',
    icon: 'fa-question-circle'
  },
  {
    id: 'trec-covid',
    name: 'TREC-COVID',
    icon: 'fa-virus'
  }
];

// Function to show selected dataset
function showDataset(datasetId, systemId) {
  // Hide all content
  document.querySelectorAll(`#tab-system-${systemId} .tab-content`).forEach(content => {
    content.classList.remove('active');
  });
  
  // Show selected content
  document.getElementById(`${datasetId}-content-${systemId}`).classList.add('active');
  
  // Update tab styling
  document.querySelectorAll(`#tab-system-${systemId} .tab-button`).forEach(button => {
    button.classList.remove('active');
  });
  document.querySelector(`#tab-system-${systemId} [onclick*="${datasetId}"]`).classList.add('active');
}

// Function to generate tabs
function generateTabs(systemId) {
  const tabsContainer = document.getElementById(`tabs-${systemId}`);
  
  datasets.forEach(dataset => {
    const button = document.createElement('button');
    button.className = `tab-button${dataset.isDefault ? ' active important' : ''}`;
    button.onclick = () => showDataset(dataset.id, systemId);
    
    button.innerHTML = `
      <i class="fas ${dataset.icon}" style="color: #666; width: 1.5em; text-align: center"></i>
      ${dataset.name}
    `;
    
    tabsContainer.appendChild(button);
  });
}

// Function to generate content
function generateContent(systemId) {
  const contentContainer = document.getElementById(`content-${systemId}`);
  
  datasets.forEach(dataset => {
    const metrics = systemId === 1 ? 
      [
        {name: 'Mean Squared Error (MSE)', metric: 'mse'},
        {name: 'Mean Absolute Error (MAE)', metric: 'mae'}
      ] :
      [
        {name: 'R2 MSE', metric: 'r2_mse'},
        {name: 'Percent of MAE Explained', metric: 'r2_mae'}
      ];

    const div = document.createElement('div');
    div.className = `tab-content${dataset.isDefault ? ' active' : ''}`;
    div.id = `${dataset.id}-content-${systemId}`;
    
    div.innerHTML = `
      <div class="dataset-grid">
        ${metrics.map(metric => `
          <figure>
            <b>${metric.name}</b>
            <iframe
              src="./figs/adriano/figs/html/${dataset.id}_olsaffine_${metric.metric}_with${systemId === 1 ? '' : 'out'}log_validation.html"
              scrolling="no"></iframe>
            <figcaption>
              ${metric.name} loss visualization across model pairs
              ${metric.metric === 'mse' ? '(Training Objective)' : ''} - ${dataset.name} Dataset
            </figcaption>
          </figure>
        `).join('')}
      </div>
    `;
    
    contentContainer.appendChild(div);
  });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Initialize both tab systems
  [1, 2].forEach(systemId => {
    generateTabs(systemId);
    generateContent(systemId);
  });
}); 

function showDataset(datasetId, systemId) {
    console.log("Running " + `tab-system-${systemId}`);
    // Get the specific tab system container
    const container = document.getElementById(`tab-system-${systemId}`);
    if (!container) {
      console.log("Container not found");
      return;
    }
    
    // Hide all content within this system
    container.querySelectorAll(".tab-content").forEach((content) => {
      content.classList.remove("active");
    });

    // Show selected content - FIXED: use querySelector instead of getElementById
    container.querySelector(`#${datasetId}-content-${systemId}`).classList.add("active");  // <--- [CHANGED] getElementById -> querySelector

    // Update button states within this system
    container.querySelectorAll(".tab-button").forEach((button) => {
      button.classList.remove("active");
    });
    container
      .querySelector(`[onclick="showDataset('${datasetId}', ${systemId})"]`)
      .classList.add("active");
  }

  function showModelView(viewId) {
    // Hide all content
    document
      .querySelectorAll(".model-content .tab-content")
      .forEach((content) => {
        content.classList.remove("active");
      });

    // Show selected content
    document.getElementById(viewId + "-content").classList.add("active");

    // Update button states
    document
      .querySelectorAll(".model-tabs .tab-button")
      .forEach((button) => {
        button.classList.remove("active");
      });
    document
      .querySelector(`.model-tabs [onclick="showModelView('${viewId}')"]`)
      .classList.add("active");
  }