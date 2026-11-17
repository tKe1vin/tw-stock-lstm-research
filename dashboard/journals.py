"""Browse retained journals and actual attempts without running daily research."""
from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import posixpath
import re
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/tKe1vin/tw-stock-lstm-research/blob/main/"
DAY = r"\d{4}-\d{2}-\d{2}"
ALLOWED = re.compile(
    rf"reports/(?:journals/(?:444|555)\.md|latest/(?:summary\.md|four_systems\.json)|"
    rf"daily/{DAY}/(?:444\.md|555\.md|summary\.md|comparison\.md|status\.json|paired/comparison\.md)|"
    rf"attempts/{DAY}/(?:summary\.md|[A-Za-z0-9_+.-]+\.json))\Z")
DAILY_FILES = {"444 日誌": "444.md", "555 日誌": "555.md",
               "主研究配對比較": "paired/comparison.md", "當日摘要": "summary.md",
               "原始比較／歷史診斷": "comparison.md", "四系統狀態": "status.json"}


class JournalError(ValueError):
    """A report is missing, malformed or outside the explicit journal contract."""


def _day(value):
    value = str(value)
    if not re.fullmatch(DAY, value):
        raise JournalError("日期格式不合法")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise JournalError("日期不合法") from exc
    return value


def source_path(relative, root=ROOT):
    """Only approved report files, with no symlink traversal even inside root."""
    relative = str(relative)
    if not ALLOWED.fullmatch(relative):
        raise JournalError("不允許的日誌來源")
    root = Path(root).resolve()
    candidate = root / relative
    if any(part.is_symlink() for part in (candidate, *candidate.parents) if part != root and part.is_relative_to(root)):
        raise JournalError("日誌來源不能透過符號連結讀取")
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise JournalError(f"尚未保存日誌來源：{relative}")
    return candidate


def read_document(relative, root=ROOT):
    path = source_path(relative, root)
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text) if path.suffix == ".json" else None
    except (OSError, UnicodeError, ValueError) as exc:
        raise JournalError(f"日誌來源無法讀取：{relative}") from exc
    if path.suffix == ".json" and not isinstance(value, dict):
        raise JournalError("狀態／嘗試紀錄必須為JSON物件")
    return {"path": str(relative), "text": text, "json": value,
            "source_url": GITHUB + quote(str(relative), safe="/")}


def _available(relative, root):
    try:
        source_path(relative, root)
    except JournalError:
        return False
    return True


def available_dates(kind="daily", root=ROOT):
    if kind not in {"daily", "attempts"}:
        raise JournalError("未知日誌類別")
    folder = Path(root) / "reports" / kind
    values = []
    if not folder.is_dir() or folder.is_symlink():
        return values
    for item in folder.iterdir():
        try:
            day = _day(item.name)
        except JournalError:
            continue
        if not item.is_dir() or item.is_symlink():
            continue
        names = DAILY_FILES.values() if kind == "daily" else [p.name for p in item.glob("*.json")] + ["summary.md"]
        if any(_available(f"reports/{kind}/{day}/{name}", root) for name in names):
            values.append(day)
    return sorted(values, reverse=True)


def daily_documents(day, root=ROOT):
    day = _day(day)
    return {label: f"reports/daily/{day}/{name}" for label, name in DAILY_FILES.items()
            if _available(f"reports/daily/{day}/{name}", root)}


def daily_context(day, root=ROOT):
    day = _day(day)
    path = f"reports/daily/{day}/status.json"
    if not _available(path, root):
        return None
    value = read_document(path, root)["json"]
    if _day(value.get("as_of_date")) != day:
        raise JournalError("報告資料日與保存目錄矛盾")
    if value.get("run_date") and _day(value["run_date"]) < day:
        raise JournalError("真正執行日不能早於資料日")
    return value


def journal_entries(text):
    """Keep each original dated entry intact; no synthetic replacement content."""
    starts = list(re.finditer(rf"^## ({DAY})(?=\s|$)", text, flags=re.MULTILINE))
    result = []
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        result.append({"date": _day(match.group(1)), "text": text[match.start():end].strip()})
    return result


def filtered_entries(entries, day=None, search=""):
    search = str(search).strip().casefold()
    return [entry for entry in entries if (day is None or entry["date"] == day)
            and (not search or search in entry["text"].casefold())]


def attempt_records(day, root=ROOT):
    day = _day(day)
    folder = Path(root) / "reports/attempts" / day
    rows = []
    for path in sorted(folder.glob("*.json"), reverse=True):
        relative = f"reports/attempts/{day}/{path.name}"
        document = read_document(relative, root)
        value = document["json"]
        if value.get("run_date") != day:
            raise JournalError("實際執行日與保存目錄矛盾")
        if value.get("as_of_date") and _day(value["as_of_date"]) > day:
            raise JournalError("實際執行日不能早於資料日")
        if value.get("started"):
            try:
                started = datetime.fromisoformat(value["started"])
            except (TypeError, ValueError) as exc:
                raise JournalError("啟動時間不合法") from exc
            if started.date().isoformat() != day:
                raise JournalError("啟動日期與實際執行目錄矛盾")
        steps = value.get("steps") or {}
        if not isinstance(steps, dict) or any(not isinstance(item, dict) for item in steps.values()):
            raise JournalError("四系統嘗試紀錄必須明確保存各步狀態")
        rows.append({"document": document, "started": value.get("started"),
                     "run_date": value["run_date"], "as_of_date": value.get("as_of_date"),
                     "status": value.get("attempt_status", "未提供"),
                     "steps": steps})
    return rows


def display_markdown(text, relative):
    """Make original relative document links open the matching GitHub source."""
    def rewrite(match):
        label, target = match.group(1), match.group(2)
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith(("#", "//")):
            return match.group(0)
        path = posixpath.normpath(posixpath.join(posixpath.dirname(relative), parsed.path))
        if path.startswith("../") or path.startswith("/"):
            return label
        url = GITHUB + quote(path, safe="/")
        if parsed.fragment:
            url += "#" + quote(parsed.fragment)
        return f"[{label}]({url})"
    return re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)", rewrite, text)


def _document_ui(document, *, key, title=None):
    import streamlit as st

    if title:
        st.subheader(title)
    st.caption("來源：" + document["path"])
    left, right = st.columns(2)
    left.link_button("在GitHub開啟原檔", document["source_url"], width="stretch")
    right.download_button("下載原始紀錄", document["text"], file_name=Path(document["path"]).name,
                          mime="application/json" if document["json"] is not None else "text/markdown",
                          key=key, width="stretch")
    if document["json"] is not None:
        st.json(document["json"])
    else:
        st.markdown(display_markdown(document["text"], document["path"]), unsafe_allow_html=False)


def _daily_ui(root):
    import streamlit as st

    dates = available_dates("daily", root)
    if not dates:
        st.info("此checkout尚未保存每日日誌。")
        return
    day = st.selectbox("資料日", dates, key="journals_daily_date")
    documents = daily_documents(day, root)
    status = daily_context(day, root)
    if status:
        a, b, c = st.columns(3)
        a.metric("報告資料日", status.get("as_of_date") or "未提供")
        b.metric("真正執行日", status.get("run_date") or "未提供")
        c.metric("444／555研究狀態", status.get("research_status") or "未提供")
    else:
        st.caption("此日期未保存四系統結構化狀態；執行時間以原文實際記載為準，不推定為資料日當晚。")
    label = st.selectbox("日誌／報告", list(documents), key="journals_daily_document")
    if label == "原始比較／歷史診斷":
        st.warning("原始比較可能採不同架構或舊成交假設；444／555主比較請使用同日的「主研究配對比較」。")
    _document_ui(read_document(documents[label], root), key="journals_daily_download")


def _cumulative_ui(root):
    import streamlit as st

    models = [model for model in ("444", "555") if _available(f"reports/journals/{model}.md", root)]
    if not models:
        st.info("此checkout尚未保存444／555累積日誌。")
        return
    model = st.selectbox("系統", models, key="journals_cumulative_model")
    document = read_document(f"reports/journals/{model}.md", root)
    entries = journal_entries(document["text"])
    dates = sorted({entry["date"] for entry in entries}, reverse=True)
    selected = st.selectbox("跳至資料日", ["全部日期", *dates], key="journals_cumulative_date")
    search = st.text_input("搜尋日誌文字", placeholder="例如：延後、新聞、2330", key="journals_search")
    matches = filtered_entries(entries, None if selected == "全部日期" else selected, search)
    st.caption(f"符合 {len(matches)}／{len(entries)} 篇；每篇保留原文與原協議，不能把不同歷史協議ROI串成每日損益。")
    st.link_button("開啟完整累積原檔", document["source_url"])
    st.download_button("下載完整累積日誌", document["text"], file_name=f"{model}_journal.md",
                       mime="text/markdown", key="journals_cumulative_download")
    if not matches:
        st.info("沒有符合此日期／文字的已保存日誌。")
    for index, entry in enumerate(matches):
        with st.expander(f"{entry['date']}｜{model}", expanded=index == 0):
            st.markdown(display_markdown(entry["text"], document["path"]), unsafe_allow_html=False)
    if not entries:
        st.markdown(display_markdown(document["text"], document["path"]), unsafe_allow_html=False)


def _attempts_ui(root):
    import streamlit as st

    dates = available_dates("attempts", root)
    if not dates:
        st.info("此checkout尚未保存實際執行紀錄。")
        return
    day = st.selectbox("實際執行日", dates, key="journals_attempt_date")
    st.caption("依真正執行日保存所有嘗試，包含失敗、等待、保留既有結果與維護；不把重跑當成新的前瞻訊號。")
    records = attempt_records(day, root)
    if records:
        st.dataframe([{"真正啟動時間": row["started"], "資料日": row["as_of_date"], "狀態": row["status"],
                       "444": row["steps"].get("444", {}).get("status", "未提供"),
                       "555": row["steps"].get("555", {}).get("status", "未提供"),
                       "555 V2": row["steps"].get("555_v2", {}).get("status", "未提供"),
                       "V2.11": row["steps"].get("v211", {}).get("status", "未提供")}
                      for row in records], hide_index=True, width="stretch")
        options = list(range(len(records)))
        index = st.selectbox("查看嘗試", options, format_func=lambda i: f"{records[i]['started'] or '時間未提供'}｜{records[i]['status']}",
                             key="journals_attempt_record")
        _document_ui(records[index]["document"], key="journals_attempt_download")
    else:
        st.info("此執行日尚未保存逐筆JSON，只能查看現有摘要。")
    summary = f"reports/attempts/{day}/summary.md"
    if _available(summary, root):
        with st.expander("此執行日的原始摘要"):
            _document_ui(read_document(summary, root), key="journals_attempt_summary_download")


def render_journals(root=ROOT):
    import streamlit as st

    st.subheader("每日日誌與執行紀錄")
    st.write("閱讀444／555的累積與逐日日誌，並追溯四系統真正執行時間、資料日及每次嘗試。")
    st.caption("555 V2與V2.11目前沒有獨立累積Markdown日誌；可在四系統狀態及實際執行紀錄查看其結果，不補造日誌。預測與研究配置不代表已下單或已實現損益。")
    mode = st.radio("瀏覽方式", ["每日報告", "累積日誌", "實際執行紀錄", "最新摘要"],
                    horizontal=True, key="journals_mode")
    try:
        if mode == "每日報告":
            _daily_ui(root)
        elif mode == "累積日誌":
            _cumulative_ui(root)
        elif mode == "實際執行紀錄":
            _attempts_ui(root)
        else:
            _document_ui(read_document("reports/latest/summary.md", root), key="journals_latest_download")
    except (JournalError, OSError, ValueError) as exc:
        st.error("日誌來源無法確認，暫停顯示此份紀錄。")
        st.caption(str(exc))
