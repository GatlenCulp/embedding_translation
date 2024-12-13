// async function loadBibliography() {
//     console.log('Starting bibliography load...'); 
//     try {
//         // Fetch the .bib file
//         const response = await fetch('bibliography.bib');
//         console.log('Fetch response:', response.status);
        
//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }
//         const bibContent = await response.text();
//         console.log('Bibliography content loaded, length:', bibContent.length);
        
//         // Create or update the bibliography script tag
//         let bibElement = document.querySelector('script[type="text/bibliography"]');
//         if (!bibElement) {
//             bibElement = document.createElement('script');
//             bibElement.type = 'text/bibliography';
//             document.body.appendChild(bibElement);
//         }
        
//         // Update content
//         bibElement.textContent = bibContent;
        
//         // Try to access Distill's internal citation system
//         if (window.distill && window.distill.template && window.distill.template.bibliography) {
//             console.log('Found Distill bibliography system, updating...');
            
//             // Parse bibliography content
//             const bibParser = new window.distill.template.Bibliography(bibContent);
//             const parsedBib = bibParser.parse();
            
//             // Update Distill's internal bibliography
//             window.distill.template.bibliography = parsedBib;
            
//             // Force update all citation elements
//             document.querySelectorAll('d-cite').forEach(cite => {
//                 const keys = cite.getAttribute('key').split(',');
//                 const references = keys.map(key => parsedBib[key.trim()]).filter(Boolean);
//                 if (references.length) {
//                     cite.references = references;
//                     // Force component update
//                     cite.requestUpdate();
//                 }
//             });
            
//             console.log('Bibliography updated');
//         } else {
//             console.warn('Distill bibliography system not found, adding front matter...');
            
//             // Create front matter if it doesn't exist
//             if (!document.querySelector('d-front-matter')) {
//                 const frontMatter = document.createElement('d-front-matter');
//                 document.body.insertBefore(frontMatter, document.body.firstChild);
//                 console.log('Added front matter element');
//             }
//         }

//         // Force article update
//         const article = document.querySelector('dt-article');
//         if (article) {
//             console.log('Found article, triggering update...');
//             // Force update by touching the DOM
//             article.style.display = 'none';
//             article.offsetHeight; // Force reflow
//             article.style.display = '';
            
//             // Also try to force Distill to reprocess
//             if (window.distill && window.distill.template) {
//                 window.distill.template.render();
//             }
//         }

//     } catch (error) {
//         console.error('Error loading bibliography:', error);
//     }
// }

// // Initial load attempts
// document.addEventListener('DOMContentLoaded', () => {
//     setTimeout(loadBibliography, 500);
// });

// window.addEventListener('load', () => {
//     setTimeout(loadBibliography, 1000);
// });

// // Add reload button for manual refresh
// const reloadButton = document.createElement('button');
// reloadButton.textContent = 'Reload Bibliography';
// reloadButton.style.position = 'fixed';
// reloadButton.style.bottom = '20px';
// reloadButton.style.right = '20px';
// reloadButton.style.zIndex = '1000';
// reloadButton.addEventListener('click', loadBibliography);
// document.body.appendChild(reloadButton);

// // Debug helper
// window.checkCitations = function() {
//     const citations = document.querySelectorAll('d-cite');
//     console.log('Found citations:', citations.length);
//     citations.forEach(citation => {
//         const key = citation.getAttribute('key');
//         const refs = citation.references;
//         console.log('Citation key:', key, 'References:', refs);
//     });
// };

// Doesn't work
