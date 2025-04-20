// Create overlay for displaying results
const overlay = document.createElement('div');
overlay.id = 'engagement-overlay';
overlay.style.position = 'fixed';
overlay.style.bottom = '20px';
overlay.style.right = '20px';
overlay.style.backgroundColor = 'rgba(0,0,0,0.7)';
overlay.style.color = 'white';
overlay.style.padding = '10px';
overlay.style.borderRadius = '5px';
overlay.style.zIndex = '10000';
overlay.style.fontFamily = 'Arial, sans-serif';
overlay.style.display = 'none';
overlay.style.maxWidth = '300px';

document.body.appendChild(overlay);

// Function to send frame to Python server
async function analyzeFrame(frameData) {
  try {
    const response = await fetch('http://localhost:5000/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: frameData })
    });
    return await response.json();
  } catch (error) {
    console.error('Analysis error:', error);
    return null;
  }
}

// Process video frames
function setupVideoAnalysis() {
  const videoElements = document.querySelectorAll('video');
  
  videoElements.forEach(video => {
    if (!video.hasAttribute('data-engagement-analyzed')) {
      video.setAttribute('data-engagement-analyzed', 'true');
      
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      let lastProcessTime = 0;
      const processInterval = 1000; // Process every 1 second (adjust as needed)

      function processFrame() {
        const now = Date.now();
        if (now - lastProcessTime < processInterval) {
          requestAnimationFrame(processFrame);
          return;
        }
        lastProcessTime = now;

        if (video.readyState >= video.HAVE_ENOUGH_DATA && video.videoWidth > 0) {
          try {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            
            analyzeFrame(frameData).then(results => {
              if (results && results.faces && results.faces.length > 0) {
                const face = results.faces[0];
                updateOverlay(face);
                
                // Send data to popup
                chrome.runtime.sendMessage({
                  action: 'updateStats',
                  data: {
                    emotion: face.emotion,
                    score: face.emotion_score,
                    engagement: face.engagement_level
                  }
                });
              } else {
                overlay.style.display = 'none';
              }
            });
          } catch (e) {
            console.error('Frame processing error:', e);
          }
        }
        requestAnimationFrame(processFrame);
      }

      video.addEventListener('play', () => {
        processFrame();
      });
    }
  });
}

function updateOverlay(faceData) {
  overlay.innerHTML = `
    <div style="margin-bottom: 5px;">
      <strong>Engagement Analysis</strong>
      <span style="float: right; background: ${getEngagementColor(faceData.engagement_level)}; 
        padding: 2px 5px; border-radius: 3px; font-size: 0.8em;">
        ${faceData.engagement_level}
      </span>
    </div>
    <div style="font-size: 0.9em;">
      <div>Emotion: ${faceData.emotion} (${(faceData.emotion_score * 100).toFixed(1)}%)</div>
      <div>Smile: ${(faceData.smile_score * 100).toFixed(0)}%</div>
    </div>
  `;
  overlay.style.display = 'block';
}

function getEngagementColor(level) {
  switch(level) {
    case 'Fully Engaged': return '#4CAF50';
    case 'Partially Engaged': return '#FFC107';
    case 'Not Engaged': return '#F44336';
    default: return '#9E9E9E';
  }
}

// Check for new video elements periodically
const observer = new MutationObserver(setupVideoAnalysis);
observer.observe(document.body, { childList: true, subtree: true });

// Initial setup
setupVideoAnalysis();

// Listen for extension toggle messages
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'toggleOverlay') {
    overlay.style.display = request.visible ? 'block' : 'none';
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'toggleAnalysis') {
    if (request.active) {
      setupVideoAnalysis();
      overlay.style.display = 'block';
    } else {
      overlay.style.display = 'none';
    }
  }
});