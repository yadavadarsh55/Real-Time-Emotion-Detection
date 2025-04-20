document.addEventListener('DOMContentLoaded', async () => {
  const toggleBtn = document.getElementById('toggle-btn');
  const settingsBtn = document.getElementById('settings-btn');
  const statusIndicator = document.getElementById('status-indicator');
  const statusText = document.getElementById('status-text');
  
  // Get current status
  const { isActive } = await chrome.storage.sync.get('isActive');
  updateStatus(isActive);
  
  // Toggle button click
  toggleBtn.addEventListener('click', async () => {
    const { isActive } = await chrome.storage.sync.get('isActive');
    const newStatus = !isActive;
    await chrome.storage.sync.set({ isActive: newStatus });
    updateStatus(newStatus);
    
    // Send message to content script
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (newStatus) {
      chrome.tabs.sendMessage(tab.id, { action: 'startAnalysis' });
    } else {
      chrome.tabs.sendMessage(tab.id, { action: 'stopAnalysis' });
    }
  });
  
  // Settings button click
  settingsBtn.addEventListener('click', () => {
    // Open options page if needed
    chrome.runtime.openOptionsPage();
  });
  
  // Update UI based on status
  function updateStatus(isActive) {
    if (isActive) {
      statusIndicator.classList.add('active');
      statusText.textContent = 'Active';
      toggleBtn.textContent = 'Deactivate';
    } else {
      statusIndicator.classList.remove('active');
      statusText.textContent = 'Inactive';
      toggleBtn.textContent = 'Activate';
    }
  }
  
  // Listen for updates from content script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'updateStats') {
      document.getElementById('current-emotion').textContent = request.data.emotion;
      document.getElementById('emotion-score').textContent = request.data.score.toFixed(2);
      document.getElementById('engagement-level').textContent = request.data.engagement;
    }
  });
});