"""Create or remove this package's own scheduled task. Nothing is registered by default.

The existing developer task TwStockLstmV211Daily is refused. This tool never
points a task at that ledger.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAME = "TwStockLstmV211Daily"
DEFAULT_NAME = "TwStockLstmCompleteDaily"


def _task_name(value: str) -> str:
    name = value.strip()
    if name == FORBIDDEN_NAME or name.lower() == FORBIDDEN_NAME.lower():
        raise SystemExit("拒絕建立 TwStockLstmV211Daily。那是開發機上既有的 V2.11 工作，不能再指向同一份帳本。")
    if not name or any(char in name for char in "\\/\"'"):
        raise SystemExit("工作名稱含有不允許的字元")
    return name


def _xml(name: str) -> str:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    script = ROOT / "scripts" / "run_all_daily.py"
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>TW Stock LSTM complete package. Separate from TwStockLstmV211Daily. Exit 2 means retry, not success.</Description>
    <URI>\\{name}</URI>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>Queue</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT4H</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2026-11-06T18:30:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <DaysOfWeek><Monday/><Tuesday/><Wednesday/><Thursday/><Friday/></DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
    <CalendarTrigger>
      <StartBoundary>2026-11-06T20:30:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <DaysOfWeek><Monday/><Tuesday/><Wednesday/><Thursday/><Friday/></DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
    <CalendarTrigger>
      <StartBoundary>2026-11-06T22:30:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <DaysOfWeek><Monday/><Tuesday/><Wednesday/><Thursday/><Friday/></DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>{python}</Command>
      <Arguments>"{script}" --mode full</Arguments>
      <WorkingDirectory>{ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--task-name", default=DEFAULT_NAME)
    args = parser.parse_args()
    name = _task_name(args.task_name)
    if args.install and args.remove:
        raise SystemExit("不能同時安裝和移除")
    if not args.install and not args.remove:
        print("預設不建立工作排程。")
        print(f"安裝：python scripts/setup_scheduler.py --install --task-name {DEFAULT_NAME}")
        print(f"移除：python scripts/setup_scheduler.py --remove --task-name {DEFAULT_NAME}")
        print("時間是台北時間週一至週五 18:30、20:30、22:30。")
        print("程式自己會跳過非交易日、特殊休市和資料未到齊的情況，並以結束代碼 2 保留重試。")
        print("此工作不會喚醒睡眠中的電腦，也需要使用者登入。它不會改到 TwStockLstmV211Daily。")
        return 0
    if args.remove:
        if name == FORBIDDEN_NAME:
            raise SystemExit("拒絕移除開發機的既有工作")
        completed = subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], text=True)
        return int(completed.returncode)
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        raise SystemExit("請先執行 install.bat，讓這個套件有自己的虛擬環境。")
    xml_path = ROOT / "logs" / "orchestrator" / f"{name}.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(_xml(name), encoding="utf-16")
    completed = subprocess.run(["schtasks", "/Create", "/TN", name, "/XML", str(xml_path), "/F"], text=True)
    if completed.returncode != 0:
        return int(completed.returncode)
    print(f"已建立 {name}。工作目錄是這個套件，不是開發專案。")
    print("它沒有指向 TwStockLstmV211Daily 的帳本。兩台電腦不會因此變成同一個發布來源。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
