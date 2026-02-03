# 📋 Phase 3: Configuration & Scanner Control - Detailed Tasks

## 🎯 Mục Tiêu Phase 3
Thêm configuration manager và scanner control vào GUI: settings panel, API key management, scanner controls, và settings persistence.

## 📌 Prerequisites
- ✅ Phase 1 đã hoàn thành (GUI Dashboard)
- ✅ Phase 2 đã hoàn thành (Trade Execution)
- ✅ ExchangeManager đang hoạt động
- ✅ Scanner module đã có

---

## ⚙️ I. CONFIGURATION PANEL

### 1.1 Create Config Panel Frame
- [x] **Task 1.1.1:** Tạo `gui/components/config_panel.py` với tabbed interface
- [x] **Task 1.1.2:** Create Risk Settings tab
- [x] **Task 1.1.3:** Create Signal Filters tab
- [x] **Task 1.1.4:** Create API Keys tab
- [x] **Task 1.1.5:** Create Default TP/SL tab

### 1.2 Risk Settings Tab
- [x] **Task 1.2.1:** Max position size input
- [x] **Task 1.2.2:** Max open positions input
- [x] **Task 1.2.3:** Max daily loss input
- [x] **Task 1.2.4:** Default leverage selector
- [x] **Task 1.2.5:** Position sizing mode (fixed/percentage)

### 1.3 Signal Filters Tab
- [x] **Task 1.3.1:** Min signal score slider
- [x] **Task 1.3.2:** Enable/disable XGBoost checkbox
- [x] **Task 1.3.3:** Symbol whitelist/blacklist
- [x] **Task 1.3.4:** Timeframe filter
- [x] **Task 1.3.5:** Min volume filter

### 1.4 API Keys Tab
- [x] **Task 1.4.1:** API key input (masked)
- [x] **Task 1.4.2:** API secret input (masked)
- [x] **Task 1.4.3:** Test connection button
- [x] **Task 1.4.4:** Exchange selector (Binance/Demo)
- [x] **Task 1.4.5:** Save credentials to .env

### 1.5 Default TP/SL Tab
- [x] **Task 1.5.1:** Default TP percentage
- [x] **Task 1.5.2:** Default SL percentage
- [x] **Task 1.5.3:** Trailing stop option
- [x] **Task 1.5.4:** TP/SL mode selector

---

## 🔍 II. SCANNER CONTROL PANEL

### 2.1 Create Scanner Control Frame
- [x] **Task 2.1.1:** Tạo `gui/components/scanner_control.py`
- [x] **Task 2.1.2:** Status indicator (running/stopped)
- [x] **Task 2.1.3:** Start/Stop buttons
- [x] **Task 2.1.4:** Last scan timestamp display
- [x] **Task 2.1.5:** Scan progress indicator

### 2.2 Scanner Configuration
- [x] **Task 2.2.1:** Scan interval input (minutes)
- [x] **Task 2.2.2:** Symbol list selector
- [x] **Task 2.2.3:** Timeframe selector
- [x] **Task 2.2.4:** Manual scan trigger button
- [x] **Task 2.2.5:** Auto-scan on startup checkbox

### 2.3 Scanner Background Loop
- [x] **Task 2.3.1:** Implement scanner thread in main_window.py
- [x] **Task 2.3.2:** Start/stop scanner on toggle
- [x] **Task 2.3.3:** Update last scan timestamp
- [x] **Task 2.3.4:** Refresh signals after scan
- [x] **Task 2.3.5:** Handle scanner errors gracefully

---

## 💾 III. SETTINGS PERSISTENCE

### 3.1 Settings Manager
- [x] **Task 3.1.1:** Tạo `gui/utils/settings_manager.py`
- [x] **Task 3.1.2:** Load settings from JSON file
- [x] **Task 3.1.3:** Save settings to JSON file
- [x] **Task 3.1.4:** Default settings fallback
- [x] **Task 3.1.5:** Settings validation

### 3.2 Settings Schema
- [x] **Task 3.2.1:** Define settings JSON schema
- [x] **Task 3.2.2:** Risk settings structure
- [x] **Task 3.2.3:** Signal filters structure
- [x] **Task 3.2.4:** Scanner settings structure
- [x] **Task 3.2.5:** UI preferences structure

### 3.3 Import/Export
- [x] **Task 3.3.1:** Export settings to file button
- [x] **Task 3.3.2:** Import settings from file button
- [x] **Task 3.3.3:** Reset to defaults button
- [x] **Task 3.3.4:** Settings backup feature
- [ ] **Task 3.3.5:** Settings migration on version upgrade

---

## 🎨 IV. UI PREFERENCES

### 4.1 Theme Settings
- [x] **Task 4.1.1:** Dark/Light mode toggle
- [x] **Task 4.1.2:** Color scheme selector
- [x] **Task 4.1.3:** Font size adjustment
- [x] **Task 4.1.4:** Apply theme without restart
- [x] **Task 4.1.5:** Save theme preference

### 4.2 Layout Preferences
- [x] **Task 4.2.1:** Remember window size/position
- [x] **Task 4.2.2:** Restore last active tab
- [x] **Task 4.2.3:** Column visibility toggles
- [x] **Task 4.2.4:** Dashboard widget order
- [x] **Task 4.2.5:** Auto-refresh intervals

---

## 🔗 V. INTEGRATION

### 5.1 Add Config Tab to Main Window
- [x] **Task 5.1.1:** Add "Settings" tab to tabview
- [x] **Task 5.1.2:** Integrate ConfigPanel
- [x] **Task 5.1.3:** Integrate ScannerControl
- [x] **Task 5.1.4:** Load settings on startup
- [x] **Task 5.1.5:** Apply settings across app

### 5.2 Settings Usage in Other Components
- [x] **Task 5.2.1:** Use risk settings in TradeForm
- [ ] **Task 5.2.2:** Use signal filters in SignalsFrame
- [x] **Task 5.2.3:** Use scanner settings in auto-trade
- [x] **Task 5.2.4:** Use theme in all components
- [x] **Task 5.2.5:** Reactive settings updates

---

## ✅ VI. TESTING

### 6.1 Config Panel Testing
- [x] Test all input fields
- [x] Test save/load functionality
- [x] Test validation
- [x] Test API key masking
- [x] Test connection test button

### 6.2 Scanner Control Testing
- [x] Test start/stop scanner
- [x] Test manual scan trigger
- [x] Test scan interval changes
- [x] Test error handling
- [x] Test UI updates

### 6.3 Settings Persistence Testing
- [x] Test save settings to file
- [x] Test load settings from file
- [x] Test import/export
- [x] Test reset to defaults
- [x] Test settings survive restart

---

## 📦 VII. DELIVERABLES

### 7.1 Code
- [x] `gui/components/config_panel.py`
- [x] `gui/components/scanner_control.py`
- [x] `gui/utils/settings_manager.py`
- [x] Updated `gui/main_window.py`

### 7.2 Features
- [x] Full configuration interface
- [x] Scanner controls working
- [x] Settings persistence
- [x] Theme customization
- [x] Import/Export settings

---

## 🎯 SUCCESS CRITERIA

Phase 3 complete when:
1. ✅ Config panel displays all settings
2. ✅ Scanner can be controlled from GUI
3. ✅ Settings save/load correctly
4. ✅ Theme changes apply
5. ✅ All tests passing

**Estimated Time:** 2-3 days  
**Priority:** MEDIUM  
**Dependencies:** Phase 1-2 Complete
