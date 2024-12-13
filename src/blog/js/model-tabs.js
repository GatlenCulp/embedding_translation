// Function to show selected model view
function showModelView(viewType) {
  // Hide all content
  document.querySelectorAll('.model-content .tab-content').forEach(content => {
    content.classList.remove('active');
  });
  
  // Show selected content
  document.getElementById(`${viewType}-content`).classList.add('active');
  
  // Update tab styling
  document.querySelectorAll('.model-tabs .tab-button').forEach(button => {
    button.classList.remove('active');
  });
  document.querySelector(`.model-tabs [onclick*="${viewType}"]`).classList.add('active');
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Set initial active tab
  showModelView('category');
}); 