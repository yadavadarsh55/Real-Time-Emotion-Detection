chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({ isActive: false });
});

chrome.action.onClicked.addListener(async (tab) => {
  const { isActive } = await chrome.storage.sync.get('isActive');
  await chrome.storage.sync.set({ isActive: !isActive });
  
  // Send toggle message to content script
  chrome.tabs.sendMessage(tab.id, {
    action: 'toggleAnalysis',
    active: !isActive
  });
  
  chrome.action.setIcon({
    tabId: tab.id,
    path: isActive ? "icons/icon16.png" : "icons/icon_active16.png"
  });
});