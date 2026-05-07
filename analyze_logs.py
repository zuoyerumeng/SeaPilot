import os
import re
from collections import defaultdict

TASK_LIST = [
    "AudioRecorderRecordAudio", "AudioRecorderRecordAudioWithFileName", "BrowserDraw", "BrowserMaze", "BrowserMultiply",
    "CameraTakePhoto", "CameraTakeVideo", "ClockStopWatchPausedVerify", "ClockStopWatchRunning", "ClockTimerEntry",
    "ContactsAddContact", "ContactsNewContactDraft", "ExpenseAddMultiple", "ExpenseAddMultipleFromGallery", "ExpenseAddMultipleFromMarkor",
    "ExpenseAddSingle", "ExpenseDeleteDuplicates", "ExpenseDeleteDuplicates2", "ExpenseDeleteMultiple", "ExpenseDeleteMultiple2",
    "ExpenseDeleteSingle", "FilesDeleteFile", "FilesMoveFile", "MarkorAddNoteHeader", "MarkorChangeNoteContent",
    "MarkorCreateFolder", "MarkorCreateNote", "MarkorCreateNoteAndSms", "MarkorCreateNoteFromClipboard", "MarkorDeleteAllNotes",
    "MarkorDeleteNewestNote", "MarkorDeleteNote", "MarkorEditNote", "MarkorMergeNotes", "MarkorMoveNote",
    "MarkorTranscribeReceipt", "MarkorTranscribeVideo", "NotesIsTodo", "NotesMeetingAttendeeCount", "NotesRecipeIngredientCount",
    "NotesTodoItemCount", "OpenAppTaskEval", "OsmAndFavorite", "OsmAndMarker", "OsmAndTrack",
    "RecipeAddMultipleRecipes", "RecipeAddMultipleRecipesFromImage", "RecipeAddMultipleRecipesFromMarkor", "RecipeAddMultipleRecipesFromMarkor2", "RecipeAddSingleRecipe",
    "RecipeDeleteDuplicateRecipes", "RecipeDeleteDuplicateRecipes2", "RecipeDeleteDuplicateRecipes3", "RecipeDeleteMultipleRecipes", "RecipeDeleteMultipleRecipesWithConstraint",
    "RecipeDeleteMultipleRecipesWithNoise", "RecipeDeleteSingleRecipe", "RecipeDeleteSingleWithRecipeWithNoise", "RetroCreatePlaylist", "RetroPlayingQueue",
    "RetroPlaylistDuration", "RetroSavePlaylist", "SaveCopyOfReceiptTaskEval", "SimpleCalendarAddOneEvent", "SimpleCalendarAddOneEventInTwoWeeks",
    "SimpleCalendarAddOneEventRelativeDay", "SimpleCalendarAddOneEventTomorrow", "SimpleCalendarAddRepeatingEvent", "SimpleCalendarAnyEventsOnDate", "SimpleCalendarDeleteEvents",
    "SimpleCalendarDeleteEventsOnRelativeDay", "SimpleCalendarDeleteOneEvent", "SimpleCalendarEventOnDateAtTime", "SimpleCalendarEventsInNextWeek", "SimpleCalendarEventsInTimeRange",
    "SimpleCalendarEventsOnDate", "SimpleCalendarFirstEventAfterStartTime", "SimpleCalendarLocationOfEvent", "SimpleCalendarNextEvent", "SimpleCalendarNextMeetingWithPerson",
    "SimpleDrawProCreateDrawing", "SimpleSmsReply", "SimpleSmsReplyMostRecent", "SimpleSmsResend", "SimpleSmsSend",
    "SimpleSmsSendClipboardContent", "SimpleSmsSendReceivedAddress", "SportsTrackerActivitiesCountForWeek", "SportsTrackerActivitiesOnDate", "SportsTrackerActivityDuration",
    "SportsTrackerLongestDistanceActivity", "SportsTrackerTotalDistanceForCategoryOverInterval", "SportsTrackerTotalDurationForCategoryThisWeek", "SystemBluetoothTurnOff", "SystemBluetoothTurnOffVerify",
    "SystemBluetoothTurnOn", "SystemBluetoothTurnOnVerify", "SystemBrightnessMax", "SystemBrightnessMaxVerify", "SystemBrightnessMin",
    "SystemBrightnessMinVerify", "SystemCopyToClipboard", "SystemWifiTurnOff", "SystemWifiTurnOffVerify", "SystemWifiTurnOn",
    "SystemWifiTurnOnVerify", "TasksCompletedTasksForDate", "TasksDueNextWeek", "TasksDueOnDate", "TasksHighPriorityTasks",
    "TasksHighPriorityTasksDueOnDate", "TasksIncompleteTasksOnDate", "TurnOffWifiAndTurnOnBluetooth", "TurnOnWifiAndOpenApp", "VlcCreatePlaylist",
    "VlcCreateTwoPlaylists"
]

def parse_value(val_str):
    val_str = val_str.strip()
    if val_str.lower().endswith('s'):
        val_str = val_str[:-1]
    
    try:
        if '.' in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str

def extract_key_values(line):
    data = {}
   
    pairs = re.findall(r'(\w+(?:\s\w+)?)=([\d\.]+[s]?)', line)
    for k, v in pairs:
        key = k.strip().replace(" ", "_") 
        data[key] = parse_value(v)
    return data

def analyze_logs(log_dir):
    missing_files = []
    model_errors = []
    valid_logs_data = []

    EXPECTED_MONITOR_MODEL = "Qwen3-VL-2B-Instruct"
    EXPECTED_EXECUTOR_MODEL = "Qwen3-4B-Instruct"


    for task_name in TASK_LIST:
        file_name = f"{task_name}.txt"
        file_path = os.path.join(log_dir, file_name)

        if not os.path.exists(file_path):
            missing_files.append(file_name)
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            monitor_valid = False
            if EXPECTED_MONITOR_MODEL in content:
                 for line in content.splitlines():
                     if "Loading Monitor" in line and EXPECTED_MONITOR_MODEL in line:
                         monitor_valid = True
                         break
            
            executor_valid = False
            if EXPECTED_EXECUTOR_MODEL in content:
                for line in content.splitlines():
                     if "Loading Executor" in line and EXPECTED_EXECUTOR_MODEL in line:
                         executor_valid = True
                         break

            if not monitor_valid:
                model_errors.append(f"{file_name} (Monitor Model Mismatch or Not Found)")
            if not executor_valid:
                model_errors.append(f"{file_name} (Executor Model Mismatch or Not Found)")

            if "AUTOMATION METRICS REPORT" in content:
                metrics_block = content.split("AUTOMATION METRICS REPORT")[-1]
                lines = metrics_block.splitlines()
                
                log_data = {}
                
                for line in lines:
                    line = line.strip()
                    if not line.startswith(">>"):
                        continue
                    
                    clean_line = line[3:]
                    
                    if "META_" in clean_line:
                        parts = clean_line.split(":", 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val = parts[1].strip()
                            try:
                                if '.' in val:
                                    val = float(val)
                                else:
                                    val = int(val)
                            except:
                                pass
                            log_data[key] = val
                    
                    elif "PERF_" in clean_line or "Total" in clean_line:
                        parts = clean_line.split(":", 1)
                        category = parts[0].strip()
                        rest = parts[1].strip()
                        kv_data = extract_key_values(rest)
                        for sub_k, sub_v in kv_data.items():
                            full_key = f"{category}_{sub_k}"
                            log_data[full_key] = sub_v
                
                if log_data:
                    valid_logs_data.append(log_data)

        except Exception as e:
            print(f"Error reading {file_name}: {e}")



    if missing_files:
        print("❌")
        print(f"{len(missing_files)} ")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("✅ ")
    print("-" * 60)


    if model_errors:
        print("❌")
        print(f"{len(model_errors)}")
        for err in model_errors:
            print(f"  - {err}")
    else:

        print("✅ ")
        print(f"   - Monitor Model : {EXPECTED_MONITOR_MODEL}")
        print(f"   - Executor Model: {EXPECTED_EXECUTOR_MODEL}")
    print("-" * 60)


    if not valid_logs_data:
        print("⚠️")
        return

    agg_stats = defaultdict(list)
    text_stats = defaultdict(lambda: defaultdict(int))

    numeric_fields = [
        "META_Final_Score", "META_Action_Steps", "META_Replan_Count",
        "PERF_Cloud_Tokens", "PERF_Cloud_Latency",
        "PERF_Monitor_Tokens", "PERF_Monitor_Latency",
        "PERF_Executor_Tokens", "PERF_Executor_Latency",
        "PERF_System_Overhead",
        "Total_All_Steps", "Total_Latency"
    ]
    
    text_fields = ["META_Agent_Type", "META_Status"]

    for entry in valid_logs_data:
        for k, v in entry.items():
            if k in numeric_fields:
                agg_stats[k].append(v)
            elif k in text_fields:
                text_stats[k][v] += 1

    file_count = len(valid_logs_data)

    print("\n")
    print("############################################################")
    print(f"## 📊 AGGREGATED STATISTICS REPORT (Files: {file_count}) ##")
    print("############################################################")
    

    for key in ["META_Agent_Type", "META_Status"]:
        counts = dict(text_stats[key])
        count_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
        print(f">> {key:<20}: {count_str}")


    for key in ["META_Final_Score", "META_Action_Steps", "META_Replan_Count"]:
        vals = agg_stats[key]
        if vals:
            _sum = sum(vals)
            _avg = _sum / len(vals)
            print(f">> {key:<20}: Sum={_sum:<8.1f} | Count={len(vals):<5} | Mean={_avg:.4f}")

    print("-" * 60)

    perf_groups = [
        ("PERF_Cloud", ["PERF_Cloud_Tokens", "PERF_Cloud_Latency"]),
        ("PERF_Monitor", ["PERF_Monitor_Tokens", "PERF_Monitor_Latency"]),
        ("PERF_Executor", ["PERF_Executor_Tokens", "PERF_Executor_Latency"]),
        ("PERF_System", ["PERF_System_Overhead"])
    ]

    for group_name, fields in perf_groups:
        output_parts = []
        for field in fields:
            vals = agg_stats[field]
            if vals:
                _sum = sum(vals)
                _count = len(vals)
                _avg = _sum / _count
                
                label = field.replace(group_name + "_", "") 
                output_parts.append(f"{label}(Count)={_count}")
                output_parts.append(f"{label}(Mean)={_avg:.4f}")
                output_parts.append(f"{label}(Sum)={_sum:.1f}")
        
        print(f">> {group_name:<20}: " + " | ".join(output_parts))

    print("-" * 60)

    total_parts = []
    if agg_stats["Total_All_Steps"]:
        vals = agg_stats["Total_All_Steps"]
        total_parts.append(f"All Steps(Count)={len(vals)}")
        total_parts.append(f"All Steps(Mean)={sum(vals)/len(vals):.2f}")
    
    if agg_stats["Total_Latency"]:
        vals = agg_stats["Total_Latency"]
        total_parts.append(f"Latency(Count)={len(vals)}")
        total_parts.append(f"Latency(Mean)={sum(vals)/len(vals):.4f}")
        total_parts.append(f"Latency(Sum)={sum(vals):.2f}s")

    print(f">> Total               : " + " | ".join(total_parts))
    print("############################################################")

if __name__ == "__main__":

    LOG_DIRECTORY = ""  

    
    if not os.path.isdir(LOG_DIRECTORY):
        print(f"'{LOG_DIRECTORY}'")
    else:
        analyze_logs(LOG_DIRECTORY)