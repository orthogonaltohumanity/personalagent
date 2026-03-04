import os
import json
import math
import random
import select
import sys
import subprocess
import threading
from datetime import datetime
from ollama import generate as ollama_generate, embed as ollama_embed, chat as ollama_chat

from config import cfg, get_molt_client, get_code_model, get_model, resolve_path
from state import state


# ── Output callbacks ─────────────────────────────────────────────────────────

_log_callback = None
_input_callback = None
_stream_callback = None


def set_log_callback(cb):
    """Set a callback for tool log messages. cb(message: str)"""
    global _log_callback
    _log_callback = cb


def set_input_callback(cb):
    """Set a callback for user input prompts. cb(prompt: str) -> str or None"""
    global _input_callback
    _input_callback = cb


def set_stream_callback(cb):
    """Set a callback for streaming model output. cb(chunk_type, text)
    chunk_type is 'thinking_start', 'thinking', 'answer_start', 'content'"""
    global _stream_callback
    _stream_callback = cb


def _log(msg):
    """Log a message via callback if set, otherwise print."""
    if _log_callback:
        _log_callback(str(msg))
    else:
        print(msg)


def _stream_generate(model, prompt, label="Generating"):
    """Stream an ollama generate call, routing output through the stream callback."""
    _log(f"{label} with {model}...")
    if _stream_callback:
        _stream_callback('answer_start', '')
    content = ''
    for chunk in ollama_generate(model=model, prompt=prompt, stream=True):
        token = chunk.get('response', '')
        if token:
            content += token
            if _stream_callback:
                _stream_callback('content', token)
            else:
                print(token, end='', flush=True)
    if not _stream_callback:
        print()
    return content


def _stream_chat(model, messages, label="Generating"):
    """Stream an ollama chat call, routing output through the stream callback."""
    _log(f"{label} with {model}...")
    if _stream_callback:
        _stream_callback('answer_start', '')
    content = ''
    thinking = ''
    for chunk in ollama_chat(model=model, messages=messages, stream=True,
                             options={'num_ctx': cfg['ollama_context_window'], 'num_predict': -1}):
        if chunk.message.thinking:
            if not thinking and _stream_callback:
                _stream_callback('thinking_start', '')
            thinking += chunk.message.thinking
            if _stream_callback:
                _stream_callback('thinking', chunk.message.thinking)
            else:
                print(chunk.message.thinking, end='', flush=True)
        elif chunk.message.content:
            if thinking and not content and _stream_callback:
                _stream_callback('answer_start', '')
            content += chunk.message.content
            if _stream_callback:
                _stream_callback('content', chunk.message.content)
            else:
                print(chunk.message.content, end='', flush=True)
    if not _stream_callback:
        print()
    return content


def _work_path(filename: str):
    """Build an absolute path under the working directory."""
    return os.path.join(state.working_directory, filename)


def _ensure_parent_dir(path: str):
    """Create parent directory for a file path when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _resolve_workdir_subpath(path: str = '.'):
    """Resolve a path under the working directory and block path traversal."""
    base = os.path.abspath(state.working_directory)
    rel = path or '.'
    candidate = os.path.abspath(os.path.join(base, rel))
    try:
        if os.path.commonpath([base, candidate]) != base:
            raise ValueError('path escapes working directory')
    except ValueError:
        raise ValueError('path escapes working directory')
    return candidate


def _load_system_prompt_text():
    """Load system prompt from configured path, tolerant of missing files."""
    try:
        with open(resolve_path(cfg.get('system_prompt_path', 'system_prompt.md')), 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""



# ── Memory ───────────────────────────────────────────────────────────────────

def list_memory_keys():
    '''Lists all memory keys'''
    return list(state.memories.keys())


def open_memory(key: str):
    '''Returns the memory stored at 'key' '''
    entry = state.memories.get(key)
    if entry is None:
        return f"Key '{key}' not found"
    entry['accessed'] = datetime.now().isoformat()
    entry['access_count'] = entry.get('access_count', 0) + 1
    state.mark_memories_dirty()
    return entry['text']


def save_memory(key: str, text: str):
    '''Saves 'text' into memory at 'key'; appends if key exists'''
    now = datetime.now().isoformat()
    if key in state.memories:
        existing = state.memories[key]['text']
        if isinstance(existing, list):
            existing.append(text)
        else:
            state.memories[key]['text'] = [existing, text]
        state.memories[key]['accessed'] = now
    else:
        state.memories[key] = {'text': text, 'created': now, 'accessed': now, 'access_count': 0}
    state.mark_memories_dirty()
    return f'Memory Saved Under {key}'


def delete_memory(key: str):
    '''Deletes the memory stored at 'key' '''
    if key in state.memories:
        del state.memories[key]
        state.mark_memories_dirty()
        return f"Memory '{key}' deleted"
    return f"Key '{key}' not found"


def edit_memory(key: str, text: str):
    '''Overwrites the text of an existing memory at 'key' (use instead of save_memory to replace rather than append)'''
    if key not in state.memories:
        return f"Key '{key}' not found"
    state.memories[key]['text'] = text
    state.memories[key]['accessed'] = datetime.now().isoformat()
    state.mark_memories_dirty()
    return f"Memory '{key}' updated"


def search_memory(query: str, top_k: int = 3):
    '''Semantic search across all memories. Returns the top-k most relevant memories with their keys.'''
    if not state.memories:
        return "No memories stored yet."
    q_emb = _embed(query)
    scored = []
    for key, entry in state.memories.items():
        text = entry['text']
        if isinstance(text, list):
            text = ' '.join(str(t) for t in text)
        else:
            text = str(text)
        mem_emb = _embed(text[:2000])
        sim = _cosine_sim(q_emb, mem_emb)
        scored.append((sim, key, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, key, text in scored[:top_k]:
        preview = text[:200] + ('...' if len(text) > 200 else '')
        results.append(f"[{key}] (sim={sim:.3f})\n{preview}")
    return "\n---\n".join(results) if results else "No memories found."


def memory_stats():
    '''Returns memory stats: total count, most/least recently accessed, and largest entries.'''
    if not state.memories:
        return "No memories stored."
    total = len(state.memories)
    entries = []
    for key, entry in state.memories.items():
        size = len(str(entry['text']))
        entries.append({'key': key, 'accessed': entry.get('accessed', ''), 'access_count': entry.get('access_count', 0), 'size': size})
    by_accessed = sorted(entries, key=lambda e: e['accessed'], reverse=True)
    by_size = sorted(entries, key=lambda e: e['size'], reverse=True)
    lines = [f"Total memories: {total}"]
    lines.append("Most recently accessed: " + ", ".join(f"{e['key']} ({e['accessed'][:19]})" for e in by_accessed[:3]))
    lines.append("Least recently accessed: " + ", ".join(f"{e['key']} ({e['accessed'][:19]})" for e in by_accessed[-3:]))
    lines.append("Largest entries: " + ", ".join(f"{e['key']} ({e['size']} chars)" for e in by_size[:3]))
    return "\n".join(lines)


def set_short_term_goal(goal: str):
    '''Changes the agent short-term goal. This goal is shown to the planner to guide task decomposition.'''
    state.short_term_goal = goal
    _log(f"Short-term goal updated: {goal}")
    return f"Short-term goal set to: {goal}"


# ── Embeddings & similarity ─────────────────────────────────────────────────

def _embed(text):
    result = ollama_embed(model=cfg['embedding_model'], input=text)
    return result['embeddings'][0]


def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ── Web / research ───────────────────────────────────────────────────────────

def search_web(text: str):
    '''Searches the web via DuckDuckGo, saves results to a file, and returns both the file path and results.'''
    from ddgs import DDGS
    results = list(DDGS().text(text, max_results=cfg['max_web_search_results']))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = f"websearch_{timestamp}.json"
    output_path = _work_path(output_name)
    _ensure_parent_dir(output_path)

    payload = {
        'query': text,
        'timestamp': datetime.now().isoformat(),
        'results': results,
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    return {
        'saved_to': output_path,
        'result_count': len(results),
        'results': results,
    }


def check_connectivity(host: str = '8.8.8.8', count: int = 1, timeout_seconds: int = 2):
    '''Checks internet connectivity by running ping (default host: 8.8.8.8). Returns online status and command output.'''
    try:
        count = max(1, int(count))
        timeout_seconds = max(1, int(timeout_seconds))
    except (TypeError, ValueError):
        return {"online": False, "error": "count and timeout_seconds must be integers"}

    cmd = ['ping', '-c', str(count), '-W', str(timeout_seconds), host]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(3, timeout_seconds * count + 2))
    except FileNotFoundError:
        return {"online": False, "error": "ping command not found", "host": host}
    except subprocess.TimeoutExpired:
        return {"online": False, "error": "ping command timed out", "host": host}

    output = (result.stdout or '') + (result.stderr or '')
    online = result.returncode == 0
    return {
        'online': online,
        'host': host,
        'count': count,
        'timeout_seconds': timeout_seconds,
        'returncode': result.returncode,
        'output': output.strip()[:2000],
    }


# ── Email ────────────────────────────────────────────────────────────────────

def _email_settings():
    """Load email settings from environment variables."""
    def _clean_env(*keys):
        for key in keys:
            value = os.environ.get(key)
            if value is None:
                continue
            cleaned = str(value).strip()
            if cleaned:
                return cleaned
        return None

    def _int_env(*keys, default):
        for key in keys:
            value = os.environ.get(key)
            if value in (None, ''):
                continue
            try:
                return int(value)
            except ValueError:
                return default
        return default

    return {
        'imap_server': _clean_env('EMAIL_IMAP_SERVER', 'IMAP_SERVER'),
        'imap_port': _int_env('EMAIL_IMAP_PORT', 'IMAP_PORT', default=993),
        'imap_security': (_clean_env('EMAIL_IMAP_SECURITY', 'IMAP_SECURITY') or 'ssl').lower(),
        'smtp_server': _clean_env('EMAIL_SMTP_SERVER', 'SMTP_SERVER'),
        'smtp_port': _int_env('EMAIL_SMTP_PORT', 'SMTP_PORT', default=587),
        'smtp_security': (_clean_env('EMAIL_SMTP_SECURITY', 'SMTP_SECURITY') or 'starttls').lower(),
        'username': _clean_env('EMAIL_USERNAME', 'EMAIL_USER', 'EMAIL_ADDRESS'),
        'password': _clean_env('EMAIL_PASSWORD', 'EMAIL_PASS'),
        'imap_username': _clean_env('EMAIL_IMAP_USERNAME', 'IMAP_USERNAME', 'EMAIL_USERNAME', 'EMAIL_USER', 'EMAIL_ADDRESS'),
        'imap_password': _clean_env('EMAIL_IMAP_PASSWORD', 'IMAP_PASSWORD', 'EMAIL_PASSWORD', 'EMAIL_PASS'),
        'smtp_username': _clean_env('EMAIL_SMTP_USERNAME', 'SMTP_USERNAME', 'EMAIL_USERNAME', 'EMAIL_USER', 'EMAIL_ADDRESS'),
        'smtp_password': _clean_env('EMAIL_SMTP_PASSWORD', 'SMTP_PASSWORD', 'EMAIL_PASSWORD', 'EMAIL_PASS'),
        'smtp_auth_mode': (_clean_env('EMAIL_SMTP_AUTH', 'SMTP_AUTH') or 'login').lower(),
        'from_address': _clean_env('EMAIL_ADDRESS', 'EMAIL_USERNAME', 'EMAIL_USER'),
    }


def _open_imap_connection(settings):
    import imaplib

    mode = (settings.get('imap_security') or 'ssl').lower()
    if mode == 'ssl':
        return imaplib.IMAP4_SSL(settings['imap_server'], settings['imap_port'])
    if mode == 'plain':
        return imaplib.IMAP4(settings['imap_server'], settings['imap_port'])
    if mode == 'starttls':
        conn = imaplib.IMAP4(settings['imap_server'], settings['imap_port'])
        conn.starttls()
        return conn
    raise ValueError('imap_security must be one of: ssl, starttls, plain')


def _decode_mime_header(value: str):
    from email.header import decode_header
    if not value:
        return ''
    parts = []
    for part, enc in decode_header(value):
        if isinstance(part, bytes):
            parts.append(part.decode(enc or 'utf-8', errors='replace'))
        else:
            parts.append(part)
    return ''.join(parts)


def _imap_disconnect(conn):
    """Best-effort IMAP close/logout without raising."""
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass
    try:
        conn.logout()
    except Exception:
        pass


def _find_imap_tuple_part(msg_data):
    """Returns the first tuple part from IMAP fetch results."""
    for part in msg_data or []:
        if isinstance(part, tuple):
            return part
    return None


def _extract_email_payload(msg):
    """Extracts text body from a message, preferring text/plain and ignoring attachments."""
    if msg.is_multipart():
        chunks = []
        for part in msg.walk():
            maintype = part.get_content_maintype()
            disp = str(part.get('Content-Disposition') or '').lower()
            if maintype != 'text' or 'attachment' in disp:
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or 'utf-8'
            try:
                chunks.append(payload.decode(charset, errors='replace'))
            except LookupError:
                chunks.append(payload.decode('utf-8', errors='replace'))
        return '\n'.join([c for c in chunks if c]).strip()

    payload = msg.get_payload(decode=True) or b''
    charset = msg.get_content_charset() or 'utf-8'
    try:
        return payload.decode(charset, errors='replace').strip()
    except LookupError:
        return payload.decode('utf-8', errors='replace').strip()


def list_emails(category: str = 'unread', mailbox: str = 'INBOX', limit: int = 10):
    '''Lists emails by category (unread/read/all) from a mailbox with sender, subject, date, and seen status.'''
    import email

    settings = _email_settings()
    required = ('imap_server', 'imap_username', 'imap_password')
    missing = [k for k in required if not settings.get(k)]
    if missing:
        return {'error': f"Missing email env vars for IMAP: {', '.join(missing)}"}

    category_value = str(category or 'unread').strip().lower()
    if category_value == 'unread':
        criteria = '(UNSEEN)'
    elif category_value == 'read':
        criteria = '(SEEN)'
    elif category_value == 'all':
        criteria = 'ALL'
    else:
        return {'error': "category must be one of: unread, read, all"}

    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        return {'error': 'limit must be an integer'}
    result_rows = []

    try:
        conn = _open_imap_connection(settings)
    except ValueError as e:
        return {'error': str(e)}
    try:
        conn.login(settings['imap_username'], settings['imap_password'])
        status, _ = conn.select(mailbox)
        if status != 'OK':
            return {'error': f"Unable to open mailbox '{mailbox}'"}

        status, data = conn.search(None, criteria)
        if status != 'OK':
            return {'error': f"Search failed for category '{category}'"}

        ids = data[0].split()[-limit:]
        ids.reverse()
        for mid in ids:
            status, msg_data = conn.fetch(mid, '(BODY.PEEK[HEADER] FLAGS)')
            if status != 'OK' or not msg_data:
                continue

            tuple_part = _find_imap_tuple_part(msg_data)
            if tuple_part is None or not isinstance(tuple_part[1], bytes):
                continue
            header_bytes = tuple_part[1]
            flags_blob = str(tuple_part[0])
            msg = email.message_from_bytes(header_bytes)
            seen = '\\Seen' in flags_blob
            result_rows.append({
                'id': mid.decode(),
                'from': _decode_mime_header(msg.get('From', '')),
                'subject': _decode_mime_header(msg.get('Subject', '')),
                'date': msg.get('Date', ''),
                'seen': seen,
            })

        return {
            'mailbox': mailbox,
            'category': category_value,
            'count': len(result_rows),
            'emails': result_rows,
        }
    finally:
        _imap_disconnect(conn)


def read_email(message_id: str, mailbox: str = 'INBOX'):
    '''Reads one email by IMAP message id and returns headers/body text.'''
    import email

    settings = _email_settings()
    required = ('imap_server', 'imap_username', 'imap_password')
    missing = [k for k in required if not settings.get(k)]
    if missing:
        return {'error': f"Missing email env vars for IMAP: {', '.join(missing)}"}

    message_id = str(message_id).strip()
    if not message_id:
        return {'error': 'message_id is required'}

    try:
        conn = _open_imap_connection(settings)
    except ValueError as e:
        return {'error': str(e)}
    try:
        conn.login(settings['imap_username'], settings['imap_password'])
        status, _ = conn.select(mailbox)
        if status != 'OK':
            return {'error': f"Unable to open mailbox '{mailbox}'"}

        status, msg_data = conn.fetch(message_id.encode(), '(RFC822 FLAGS)')
        if status != 'OK' or not msg_data:
            return {'error': f"Unable to fetch email id {message_id}"}

        tuple_part = _find_imap_tuple_part(msg_data)
        if tuple_part is None or not isinstance(tuple_part[1], bytes):
            return {'error': f"No message payload for id {message_id}"}
        raw_msg = tuple_part[1]
        flags_blob = str(tuple_part[0])

        msg = email.message_from_bytes(raw_msg)
        body = _extract_email_payload(msg)

        return {
            'id': message_id,
            'mailbox': mailbox,
            'from': _decode_mime_header(msg.get('From', '')),
            'to': _decode_mime_header(msg.get('To', '')),
            'subject': _decode_mime_header(msg.get('Subject', '')),
            'date': msg.get('Date', ''),
            'seen': '\\Seen' in flags_blob,
            'body': body[:20000],
        }
    finally:
        _imap_disconnect(conn)


def send_email(to: str, subject: str, body: str, cc: str = '', bcc: str = ''):
    '''Sends an email using SMTP environment configuration. cc and bcc are optional comma-separated lists.'''
    import smtplib
    from email.message import EmailMessage

    settings = _email_settings()
    required = ('smtp_server', 'from_address')
    missing = [k for k in required if not settings.get(k)]
    if missing:
        return {'error': f"Missing email env vars for SMTP: {', '.join(missing)}"}

    msg = EmailMessage()
    msg['From'] = settings['from_address']
    msg['To'] = to
    msg['Subject'] = subject
    if cc.strip():
        msg['Cc'] = cc
    msg.set_content(body)

    recipients = [e.strip() for e in (to + ',' + cc + ',' + bcc).split(',') if e.strip()]
    if not recipients:
        return {'error': 'At least one recipient is required'}

    smtp_security = (settings.get('smtp_security') or 'starttls').lower()
    if smtp_security == 'ssl':
        server_context = smtplib.SMTP_SSL(settings['smtp_server'], settings['smtp_port'], timeout=30)
    elif smtp_security in ('starttls', 'plain'):
        server_context = smtplib.SMTP(settings['smtp_server'], settings['smtp_port'], timeout=30)
    else:
        return {'error': 'smtp_security must be one of: starttls, ssl, plain'}

    smtp_auth_mode = (settings.get('smtp_auth_mode') or 'login').lower()
    if smtp_auth_mode not in ('login', 'none'):
        return {'error': 'smtp_auth_mode must be one of: login, none'}

    with server_context as server:
        server.ehlo()
        if smtp_security == 'starttls':
            server.starttls()
            server.ehlo()
        if smtp_auth_mode == 'login':
            if not settings.get('smtp_username') or not settings.get('smtp_password'):
                return {'error': 'Missing email env vars for SMTP auth: smtp_username, smtp_password'}
            server.login(settings['smtp_username'], settings['smtp_password'])
        server.send_message(msg, from_addr=settings['from_address'], to_addrs=recipients)

    return {
        'status': 'sent',
        'to': to,
        'cc': cc,
        'bcc_count': len([e for e in bcc.split(',') if e.strip()]),
        'subject': subject,
        'smtp_security': smtp_security,
        'smtp_auth_mode': smtp_auth_mode,
    }


def send_email_from_file(to: str, subject: str, source_filename: str, cc: str = '', bcc: str = ''):
    '''Reads body text from source_filename in the working directory and sends it as an email.'''
    source_path = _work_path(source_filename)
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            body = f.read()
    except FileNotFoundError:
        return {'error': f"Source file '{source_filename}' not found in {state.working_directory}"}
    except OSError as e:
        return {'error': f"Unable to read source file '{source_filename}': {e}"}

    if not body.strip():
        return {'error': f"Source file '{source_filename}' is empty"}

    result = send_email(to=to, subject=subject, body=body, cc=cc, bcc=bcc)
    if isinstance(result, dict) and result.get('status') == 'sent':
        result = dict(result)
        result['source'] = source_filename
    return result


def mark_email_seen(message_id: str, seen: bool = True, mailbox: str = 'INBOX'):
    '''Marks an email as seen/unseen by IMAP message id.'''
    settings = _email_settings()
    required = ('imap_server', 'imap_username', 'imap_password')
    missing = [k for k in required if not settings.get(k)]
    if missing:
        return {'error': f"Missing email env vars for IMAP: {', '.join(missing)}"}

    message_id = str(message_id).strip()
    if not message_id:
        return {'error': 'message_id is required'}

    try:
        conn = _open_imap_connection(settings)
    except ValueError as e:
        return {'error': str(e)}
    try:
        conn.login(settings['imap_username'], settings['imap_password'])
        status, _ = conn.select(mailbox)
        if status != 'OK':
            return {'error': f"Unable to open mailbox '{mailbox}'"}

        op = '+FLAGS' if seen else '-FLAGS'
        status, _ = conn.store(message_id.encode(), op, '(\\Seen)')
        if status != 'OK':
            return {'error': f"Unable to update seen flag for id {message_id}"}
        return {'status': 'ok', 'id': message_id, 'seen': seen, 'mailbox': mailbox}
    finally:
        _imap_disconnect(conn)



_SUPPORTED_FILETYPES = {
    "pdf": {"extensions": [".pdf"], "content_types": ["application/pdf"]},
    "csv": {"extensions": [".csv"], "content_types": ["text/csv", "application/csv", "text/plain"]},
}

_INGEST_FUNCTIONS = {}


def search_and_download_files(query: str, filetype: str = "pdf"):
    '''Searches for and downloads files of 'filetype' (pdf, csv) relevant to 'query'. Auto-ingests into the vector index.'''
    if filetype not in _SUPPORTED_FILETYPES:
        return f"Unsupported filetype '{filetype}'. Supported: {list(_SUPPORTED_FILETYPES.keys())}"
    ft = _SUPPORTED_FILETYPES[filetype]
    output_dir = str(resolve_path(cfg['downloads_directory']))
    os.makedirs(output_dir, exist_ok=True)
    from ddgs import DDGS
    import requests
    with DDGS() as ddgs:
        results = list(ddgs.text(f"{query} filetype:{filetype}", max_results=cfg['max_download_search_results']))
    downloaded = 0
    name_hist = []
    for result in results:
        url = result["href"]
        if not any(url.lower().endswith(ext) for ext in ft["extensions"]):
            continue
        try:
            resp = requests.get(url, timeout=cfg['download_timeout_seconds'], headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if not any(ct in ctype for ct in ft["content_types"]):
                continue
            filename = os.path.join(output_dir, f"{random.randrange(0, 10000) + 1}.{filetype}")
            name_hist.append(filename)
            with open(filename, "wb") as f:
                f.write(resp.content)
            _log(f"Downloaded: {url} -> {filename}")
            downloaded += 1
            ingest_fn = _INGEST_FUNCTIONS.get(filetype)
            if ingest_fn:
                try:
                    ingest_fn(filename)
                except Exception as ie:
                    _log(f"Ingest failed for {filename}: {ie}")
        except Exception as e:
            _log(f"Failed: {url} — {e}")
    _log(f"Done. {downloaded} {filetype.upper()} files saved and ingested to '{output_dir}/'")
    return f"New files: {name_hist}"


# ── PDF / CSV ingestion ──────────────────────────────────────────────────────

def ingest_pdf(filename: str):
    '''Extracts text from a PDF, chunks it, embeds each chunk, and stores in the vector index.'''
    from pypdf import PdfReader
    reader = PdfReader(filename)
    total_pages = len(reader.pages)

    if total_pages <= 10:
        chunk_size, overlap, page_step = 300, 50, 1
    elif total_pages <= 50:
        chunk_size, overlap, page_step = 600, 80, 1
    elif total_pages <= 150:
        chunk_size, overlap, page_step = 1000, 100, 2
    else:
        chunk_size, overlap, page_step = 1500, 150, 3

    count = 0
    for page_num in range(0, total_pages, page_step):
        block = ""
        end_page = min(page_num + page_step, total_pages)
        for p in range(page_num, end_page):
            block += (reader.pages[p].extract_text() or "") + "\n"
        if not block.strip():
            continue
        for chunk in _chunk_text(block, size=chunk_size, overlap=overlap):
            embedding = _embed(chunk)
            state.pdf_index.append({
                "text": chunk,
                "source": filename,
                "page": page_num + 1,
                "embedding": embedding,
            })
            count += 1
    state.save_pdf_index()
    _log(f"Ingested {filename}: {count} chunks ({total_pages} pages, step={page_step}, chunk_size={chunk_size})")
    return f"Ingested {count} chunks from {filename} ({total_pages} pages)"


def _csv_to_text(filepath, max_rows=500):
    import csv
    with open(filepath, 'r', newline='', errors='replace') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ""
    headers = rows[0]
    lines = [f"Columns: {', '.join(headers)}"]
    data_rows = rows[1:max_rows + 1]
    for i, row in enumerate(data_rows, 1):
        pairs = ", ".join(f"{h}={v}" for h, v in zip(headers, row))
        lines.append(f"Row {i}: {pairs}")
    if len(rows) - 1 > max_rows:
        lines.append(f"... ({len(rows) - 1 - max_rows} more rows truncated)")
    return "\n".join(lines)


def ingest_csv(filename: str):
    '''Reads a CSV file, converts it to readable text, chunks, embeds, and stores in the vector index.'''
    text = _csv_to_text(filename)
    if not text:
        return f"CSV '{filename}' is empty or unreadable."
    total_rows = text.count("\nRow ")
    if total_rows <= 50:
        chunk_size, overlap = 400, 50
    elif total_rows <= 200:
        chunk_size, overlap = 800, 100
    else:
        chunk_size, overlap = 1200, 150
    count = 0
    for chunk in _chunk_text(text, size=chunk_size, overlap=overlap):
        embedding = _embed(chunk)
        state.pdf_index.append({
            "text": chunk,
            "source": filename,
            "page": 0,
            "embedding": embedding,
        })
        count += 1
    state.save_pdf_index()
    _log(f"Ingested {filename}: {count} chunks ({total_rows} rows)")
    return f"Ingested {count} chunks from {filename} ({total_rows} rows)"


_INGEST_FUNCTIONS["pdf"] = ingest_pdf
_INGEST_FUNCTIONS["csv"] = ingest_csv


def query_documents(query: str, top_k: int = 5):
    '''Semantic search across all ingested documents (PDFs, CSVs). Returns the top-k most relevant chunks with source info.'''
    if not state.pdf_index:
        return "No documents ingested yet. Use ingest_pdf, ingest_csv, or search_and_download_files first."
    q_emb = _embed(query)
    scored = []
    for entry in state.pdf_index:
        sim = _cosine_sim(q_emb, entry["embedding"])
        scored.append((sim, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, entry in scored[:top_k]:
        src = entry['source']
        page = entry['page']
        loc = f"{src} p.{page}" if page > 0 else src
        results.append(f"[{loc}] (sim={sim:.3f})\n{entry['text']}")
    return "\n---\n".join(results)


def list_downloaded_files():
    '''Lists all downloaded files (PDFs, CSVs) in the downloads and pdfs directories.'''
    supported_ext = ('.pdf', '.csv')
    found = []
    for directory in (str(resolve_path(cfg['downloads_directory'])), str(resolve_path('pdfs'))):
        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith(supported_ext):
                    full_path = os.path.join(directory, filename)
                    if os.path.isfile(full_path):
                        found.append(full_path)
        except OSError:
            continue
    if not found:
        _log("No downloaded files found")
    else:
        _log(str(found))
    return found


# ── File system ──────────────────────────────────────────────────────────────

def read_file(file: str):
    '''Reads a file from the working directory'''
    try:
        with open(_work_path(file), 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def edit(file: str, append: bool, text: str):
    '''Writes 'text' to a file in the working directory; append=True to append'''
    mode = 'a' if append else 'w'
    try:
        filepath = _work_path(file)
        _ensure_parent_dir(filepath)
        with open(filepath, mode) as f:
            f.write(text)
        action = 'Appended to' if append else 'Wrote'
        return f'{action} {file} ({len(text)} chars)'
    except Exception as e:
        return f"Error: {e}"


def list_working_files(path: str = '.', recursive: bool = True):
    '''Lists files in the Playground working directory or one of its subdirectories.
    path is relative to the working directory. Set recursive=False for top-level only.'''
    try:
        root = _resolve_workdir_subpath(path)
    except Exception as e:
        return f"Error: {e}"

    if not os.path.exists(root):
        return f"Error: path '{path}' does not exist in {state.working_directory}"

    files = []
    base = os.path.abspath(state.working_directory)
    if os.path.isfile(root):
        files.append(os.path.relpath(root, base))
    else:
        if recursive:
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    full = os.path.join(dirpath, filename)
                    files.append(os.path.relpath(full, base))
        else:
            for name in os.listdir(root):
                full = os.path.join(root, name)
                if os.path.isfile(full):
                    files.append(os.path.relpath(full, base))

    files.sort()
    return files


# ── Dynamic tool creation ────────────────────────────────────────────────────

CUSTOM_TOOLS_PATH = str(resolve_path(cfg.get('custom_tools_path', 'custom_tools.json')))
TOOL_APPROVAL_TIMEOUT = cfg.get('tool_approval_timeout', 60)

_custom_tool_registry = []  # in-memory list of {name, description, code}

# These will be set by main.py after tool registry is built
available_functions = {}


def _timed_input(prompt, timeout=TOOL_APPROVAL_TIMEOUT):
    if _input_callback:
        return _input_callback(prompt)
    print(prompt, end='', flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip().lower()
    _log(f"[No response in {timeout}s — defaulting to no]")
    return None


def load_custom_tools():
    '''Load persisted custom tools from disk and register them.'''
    if not os.path.exists(CUSTOM_TOOLS_PATH):
        return
    with open(CUSTOM_TOOLS_PATH, 'r') as f:
        saved = json.load(f)
    for entry in saved:
        try:
            exec_globals = {}
            exec(entry['code'], exec_globals)
            fn = exec_globals[entry['name']]
            available_functions[entry['name']] = fn
            _custom_tool_registry.append(entry)
            _log(f"[startup] Loaded custom tool: {entry['name']}")
        except Exception as e:
            _log(f"[startup] Failed to load custom tool '{entry['name']}': {e}")


def _save_custom_tools():
    saved = [{'name': e['name'], 'description': e['description'], 'code': e['code']} for e in _custom_tool_registry]
    _ensure_parent_dir(CUSTOM_TOOLS_PATH)
    with open(CUSTOM_TOOLS_PATH, 'w') as f:
        json.dump(saved, f, indent=2)


def create_tool(function_name: str, description: str, code: str):
    '''
    Creates a new tool from Python code and registers it (with user approval).
    'function_name' must match the def in 'code'. 'description' explains what it does.
    'code' is the full Python function definition. User will be shown the code and asked to approve.
    '''
    if f'def {function_name}(' not in code:
        return f"Error: code must contain 'def {function_name}(...)'. Got code that doesn't match."

    _log(f"{'='*60}")
    _log(f"  NEW TOOL REQUEST: {function_name}")
    _log(f"  Description: {description}")
    _log(f"{'='*60}")
    _log(code)
    _log(f"{'='*60}")

    approval = _timed_input(f"Approve adding '{function_name}' as a tool? (y/n, {TOOL_APPROVAL_TIMEOUT}s timeout): ")
    if approval != 'y':
        return f"User denied creation of tool '{function_name}'."

    try:
        exec_globals = {}
        exec(code, exec_globals)
        fn = exec_globals.get(function_name)
        if fn is None or not callable(fn):
            return f"Error: after exec, '{function_name}' was not found as a callable."
        available_functions[function_name] = fn
        _custom_tool_registry.append({'name': function_name, 'description': description, 'code': code})
        _save_custom_tools()
        return f"Tool '{function_name}' created and registered successfully. It is now available for use."
    except Exception as e:
        return f"Error creating tool: {e}"


def list_custom_tools():
    '''Lists all user-approved custom tools that have been dynamically created.'''
    if not _custom_tool_registry:
        return "No custom tools have been created yet."
    lines = [f"  {entry['name']}: {entry['description']}" for entry in _custom_tool_registry]
    return "Custom tools:\n" + "\n".join(lines)


def remove_custom_tool(function_name: str):
    '''Removes a custom tool by name (requires user approval)'''
    if function_name not in available_functions:
        return f"Tool '{function_name}' not found."
    is_custom = any(e['name'] == function_name for e in _custom_tool_registry)
    if not is_custom:
        return f"'{function_name}' is a built-in tool and cannot be removed."

    approval = _timed_input(f"Approve removing custom tool '{function_name}'? (y/n, {TOOL_APPROVAL_TIMEOUT}s timeout): ")
    if approval != 'y':
        return f"User denied removal of tool '{function_name}'."

    del available_functions[function_name]
    _custom_tool_registry[:] = [e for e in _custom_tool_registry if e['name'] != function_name]
    _save_custom_tools()
    return f"Custom tool '{function_name}' removed."


# ── Code generation ──────────────────────────────────────────────────────────

def _extract_code_from_response(text):
    if '```' not in text:
        return text
    lines = text.split('\n')
    code_lines = []
    in_block = False
    for line in lines:
        if line.strip().startswith('```'):
            in_block = not in_block
            continue
        if in_block:
            code_lines.append(line)
    return '\n'.join(code_lines) if code_lines else text


def write_text(filename: str, prompt: str):
    '''Generates written text (articles, posts, essays, documentation, creative writing) using the planner model with memory context, and saves it to a file in the working directory. Use this for any non-code text generation.'''
    model = get_model('planner')

    # Build memory context from semantic search
    memory_context = ""
    if state.memories:
        mem_results = search_memory(prompt, top_k=3)
        if mem_results and "No memories" not in mem_results:
            memory_context = f"\n\nRelevant memories:\n{mem_results}\n"

    # Load system prompt
    system_prompt = _load_system_prompt_text()

    messages = [
        {'role': 'system', 'content': (
            f"{system_prompt}\n\n"
            f"You are a writer. Produce the requested text. Output ONLY the text content, no commentary."
            f"{memory_context}"
        )},
        {'role': 'user', 'content': prompt}
    ]

    timeout = cfg.get('download_subtask_timeout_seconds', 900)
    result_box = [None]
    error_box = [None]

    def run():
        try:
            result_box[0] = _stream_chat(model, messages, label=f"Writing {filename}")
        except Exception as e:
            error_box[0] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        _log(f"write_text timed out after {timeout}s")
        return f"Error: write_text timed out after {timeout}s"

    if error_box[0]:
        return f"Error: {error_box[0]}"

    text = result_box[0] or ""

    filepath = _work_path(filename)
    _ensure_parent_dir(filepath)
    with open(filepath, 'w') as f:
        f.write(text)
    return f"Generated with {model} (memory-aware) and saved to {filepath} ({len(text)} chars)"


def edit_text(filename: str, prompt: str):
    '''Reads an existing text file from the working directory, sends it with editing instructions to the planner model with memory context, and overwrites the file. Use this to revise, expand, rewrite, or otherwise modify existing non-code text.'''
    filepath = _work_path(filename)
    try:
        with open(filepath, 'r') as f:
            existing = f.read()
    except FileNotFoundError:
        return f"Error: File '{filename}' not found in {state.working_directory}"

    model = get_model('planner')

    # Build memory context from semantic search
    memory_context = ""
    if state.memories:
        mem_results = search_memory(prompt, top_k=3)
        if mem_results and "No memories" not in mem_results:
            memory_context = f"\n\nRelevant memories:\n{mem_results}\n"

    # Load system prompt
    system_prompt = _load_system_prompt_text()

    messages = [
        {'role': 'system', 'content': (
            f"{system_prompt}\n\n"
            f"You are a writer and editor. Revise the existing text according to the instructions. "
            f"Output ONLY the complete revised text, no commentary."
            f"{memory_context}"
        )},
        {'role': 'user', 'content': (
            f"Here is the existing text:\n\n---\n{existing}\n---\n\n"
            f"Instructions: {prompt}\n\n"
            f"Return the complete revised text."
        )}
    ]

    timeout = cfg.get('download_subtask_timeout_seconds', 900)
    result_box = [None]
    error_box = [None]

    def run():
        try:
            result_box[0] = _stream_chat(model, messages, label=f"Editing {filename}")
        except Exception as e:
            error_box[0] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        _log(f"edit_text timed out after {timeout}s")
        return f"Error: edit_text timed out after {timeout}s"

    if error_box[0]:
        return f"Error: {error_box[0]}"

    text = result_box[0] or ""

    with open(filepath, 'w') as f:
        f.write(text)
    return f"Edited with {model} (memory-aware) and saved to {filepath} ({len(text)} chars)"


def write_text_from_source(filename: str, source_filename: str, prompt: str):
    '''Reads a source file as reference material and generates NEW text in a separate output file. Essential for multi-stage writing pipelines: outline → draft → polished work. Use this to build upon existing text — e.g. turn an outline into a full draft, expand notes into an article, transform bullet points into prose, or write a response based on source material. The source file is NOT modified.'''
    source_path = _work_path(source_filename)
    try:
        with open(source_path, 'r') as f:
            source_content = f.read()
    except FileNotFoundError:
        return f"Error: Source file '{source_filename}' not found in {state.working_directory}"

    if not source_content.strip():
        return f"Error: Source file '{source_filename}' is empty"

    model = get_model('planner')

    # Build memory context from semantic search
    memory_context = ""
    if state.memories:
        mem_results = search_memory(prompt, top_k=3)
        if mem_results and "No memories" not in mem_results:
            memory_context = f"\n\nRelevant memories:\n{mem_results}\n"

    # Load system prompt
    system_prompt = _load_system_prompt_text()

    messages = [
        {'role': 'system', 'content': (
            f"{system_prompt}\n\n"
            f"You are a writer. Use the provided source material as context to produce the requested text. "
            f"Output ONLY the text content, no commentary."
            f"{memory_context}"
        )},
        {'role': 'user', 'content': (
            f"Source material (from {source_filename}):\n\n---\n{source_content}\n---\n\n"
            f"Instructions: {prompt}"
        )}
    ]

    timeout = cfg.get('download_subtask_timeout_seconds', 900)
    result_box = [None]
    error_box = [None]

    def run():
        try:
            result_box[0] = _stream_chat(model, messages, label=f"Writing {filename} from {source_filename}")
        except Exception as e:
            error_box[0] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        _log(f"write_text_from_source timed out after {timeout}s")
        return f"Error: write_text_from_source timed out after {timeout}s"

    if error_box[0]:
        return f"Error: {error_box[0]}"

    text = result_box[0] or ""

    filepath = _work_path(filename)
    _ensure_parent_dir(filepath)
    with open(filepath, 'w') as f:
        f.write(text)
    return f"Generated with {model} (memory-aware, source: {source_filename}) and saved to {filepath} ({len(text)} chars)"


def generate_code(filename: str, prompt: str):
    '''Generates code using the configured code model and saves it to a file in the working directory. Does NOT execute the code.'''
    code_model = get_code_model()
    raw = _stream_generate(code_model, prompt, label=f"Generating code for {filename}")
    text = _extract_code_from_response(raw)
    filepath = _work_path(filename)
    _ensure_parent_dir(filepath)
    with open(filepath, 'w') as f:
        f.write(text)
    return f"Generated with {code_model} and saved to {filepath} ({len(text)} chars)"


def generate_code_edit(filename: str, prompt: str):
    '''Reads an existing file from the working directory, sends it with 'prompt' to the code model for modification, and overwrites the file. Does NOT execute the code.'''
    filepath = _work_path(filename)
    try:
        with open(filepath, 'r') as f:
            existing = f.read()
    except FileNotFoundError:
        return f"File '{filename}' not found in {state.working_directory}"
    code_model = get_code_model()
    full_prompt = f"Here is the existing code:\n\n```\n{existing}\n```\n\n{prompt}\n\nReturn ONLY the complete modified code, no explanations."
    raw = _stream_generate(code_model, full_prompt, label=f"Editing {filename}")
    text = _extract_code_from_response(raw)
    with open(filepath, 'w') as f:
        f.write(text)
    return f"Modified with {code_model} and saved to {filepath} ({len(text)} chars)"


# ── Git (Playground only) ────────────────────────────────────────────────────

def _git(args):
    result = subprocess.run(
        ['git'] + args,
        cwd=str(state.working_directory),
        capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip()


def git_init():
    '''Initializes a git repo in the working directory'''
    return _git(['init'])


def git_status():
    '''Shows git status of the working directory'''
    return _git(['status', '--short'])


def git_add(path: str):
    '''Stages a file or directory for commit (relative to working directory)'''
    return _git(['add', path])


def git_commit(message: str):
    '''Creates a git commit with the given message'''
    return _git(['commit', '-m', message])


def git_log():
    '''Shows recent git log (last 10 commits, oneline)'''
    return _git(['log', '--oneline', '-10'])


def git_diff():
    '''Shows the current diff of unstaged changes'''
    return _git(['diff'])


def git_diff_staged():
    '''Shows the diff of staged changes'''
    return _git(['diff', '--staged'])


def git_branch(name: str):
    '''Creates a new git branch'''
    return _git(['branch', name])


def git_checkout(target: str):
    '''Switches to a branch or commit'''
    return _git(['checkout', target])


def git_list_branches():
    '''Lists all git branches'''
    return _git(['branch', '-a'])


# ── Social media (Moltbook) ─────────────────────────────────────────────────

def _moltbook_get(path):
    import urllib.request
    req = urllib.request.Request(
        f"https://www.moltbook.com/api/v1{path}",
        headers={
            "Authorization": f"Bearer {os.environ.get('MOLTBOOK_API_KEY')}",
            "Content-Type": "application/json"
        }
    )
    return json.loads(urllib.request.urlopen(req).read())


def _moltbook_post(path, data):
    import urllib.request
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"https://www.moltbook.com/api/v1{path}", data=body,
        headers={
            "Authorization": f"Bearer {os.environ.get('MOLTBOOK_API_KEY')}",
            "Content-Type": "application/json"
        }
    )
    return json.loads(urllib.request.urlopen(req).read())


def read_social_media_feed(sort: str):
    '''Reads the Moltbook feed; sort can be "hot", "new", "top" etc.'''
    try:
        return get_molt_client().feed(sort=sort, limit=5)
    except Exception as e:
        return e


def create_social_media_post(community: str, title: str, body: str):
    '''Creates a post with 'title' and 'body' in 'community' '''
    try:
        return get_molt_client().create_post(community, title, body)
    except Exception as e:
        return e


def post_file_to_social_media(community: str, title: str, filename: str):
    '''Reads a file from the working directory and posts its contents to a Moltbook community. The file contents become the post body.'''
    filepath = _work_path(filename)
    try:
        with open(filepath, 'r') as f:
            body = f.read()
    except FileNotFoundError:
        return f"Error: File '{filename}' not found in {state.working_directory}"
    if not body.strip():
        return f"Error: File '{filename}' is empty"
    try:
        return get_molt_client().create_post(community, title, body)
    except Exception as e:
        return f"Error: {e}"


def create_social_media_comment(post_uuid: str, body: str, parent_id: str = None):
    '''Creates a comment on post_uuid. Set parent_id to reply to a specific comment.'''
    try:
        return get_molt_client().comment(post_uuid, body, parent_id)
    except Exception as e:
        return e


def social_media_upvote(post_uuid: str):
    '''Upvotes the post with id 'post_uuid' '''
    try:
        return get_molt_client().upvote(post_uuid)
    except Exception as e:
        return e


def social_media_downvote(post_uuid: str):
    '''Downvotes the post with id 'post_uuid' '''
    try:
        return get_molt_client().downvote(post_uuid)
    except Exception as e:
        return e


def social_media_upvote_comment(comment_uuid: str):
    '''Upvotes the comment with id 'comment_uuid' '''
    try:
        return get_molt_client().upvote_comment(comment_uuid)
    except Exception as e:
        return e


def get_social_media_post(post_uuid: str):
    '''Fetches a single post and its comments by 'post_uuid' '''
    try:
        return get_molt_client().post(post_uuid)
    except Exception as e:
        return e


def list_community_posts(community: str, sort: str = 'hot', limit: int = 10):
    '''Lists posts from a specific community; sort: "hot"/"new"/"top" '''
    try:
        return get_molt_client().posts(community, sort=sort, limit=limit)
    except Exception as e:
        return e


def social_media_search(query: str):
    '''Searches Moltbook for posts matching 'query' '''
    try:
        return get_molt_client().search(query)
    except Exception as e:
        return e


def get_personal_history():
    '''Returns own Moltbook profile and stats'''
    try:
        return get_molt_client().me()
    except Exception as e:
        return e


def get_user_profile(username: str):
    '''Fetches a Moltbook user profile. For your own profile use check_agent_status instead.'''
    try:
        return _moltbook_get(f"/agents/{username}")
    except Exception as e:
        return e


def list_communities():
    '''Lists all available Moltbook communities/submolts'''
    try:
        return get_molt_client().submolts()
    except Exception as e:
        return e


def check_agent_status():
    '''Checks your own Moltbook account: name, karma, unread notifications, recent activity on your posts, and suggested actions'''
    try:
        return _moltbook_get("/home")
    except Exception as e:
        return e


def update_profile(description: str):
    '''Updates the agent's Moltbook profile bio/description'''
    try:
        return get_molt_client().update_profile(description)
    except Exception as e:
        return e


def make_community(name: str, display_name: str, description: str):
    '''Creates a new Moltbook community with 'name', 'display_name', and 'description' '''
    try:
        _moltbook_post("/submolts", {
            "name": name,
            "display_name": display_name,
            "description": description
        })
        return "SUCCESSFULLY MADE SUBMOLT"
    except Exception as e:
        return e


# ── Check in ─────────────────────────────────────────────────────────────────

def check_in(question: str):
    '''Asks the user a question directly and waits for their response. Use this when you need advice, clarification, or approval before proceeding.'''
    _log(f"{'='*60}")
    _log(f"AGENT CHECK-IN: {question}")
    _log(f"{'='*60}")
    response = _timed_input("Your response: ", timeout=60)
    if response is None:
        return "[No response from user within 60 seconds]"
    return response


# ── Tool registry ────────────────────────────────────────────────────────────

def build_tool_registry():
    """Build and return the available_functions dict (flat dict of all tools)."""
    available_functions.update({
        # Memory
        'list_memory_keys': list_memory_keys,
        'open_memory': open_memory,
        'save_memory': save_memory,
        'delete_memory': delete_memory,
        'edit_memory': edit_memory,
        'search_memory': search_memory,
        'memory_stats': memory_stats,
        'set_short_term_goal': set_short_term_goal,
        # Web / research
        'search_web': search_web,
        'check_connectivity': check_connectivity,
        'search_and_download_files': search_and_download_files,
        # Email
        'list_emails': list_emails,
        'read_email': read_email,
        'send_email': send_email,
        'send_email_from_file': send_email_from_file,
        'mark_email_seen': mark_email_seen,
        # Documents
        'ingest_pdf': ingest_pdf,
        'ingest_csv': ingest_csv,
        'query_documents': query_documents,
        'list_downloaded_files': list_downloaded_files,
        # Social media
        'read_social_media_feed': read_social_media_feed,
        'create_social_media_post': create_social_media_post,
        'post_file_to_social_media': post_file_to_social_media,
        'create_social_media_comment': create_social_media_comment,
        'social_media_upvote': social_media_upvote,
        'social_media_downvote': social_media_downvote,
        'social_media_upvote_comment': social_media_upvote_comment,
        'get_social_media_post': get_social_media_post,
        'list_community_posts': list_community_posts,
        'social_media_search': social_media_search,
        'get_personal_history': get_personal_history,
        'get_user_profile': get_user_profile,
        'list_communities': list_communities,
        'check_agent_status': check_agent_status,
        'update_profile': update_profile,
        'make_community': make_community,
        # Files
        'read_file': read_file,
        'edit': edit,
        'list_working_files': list_working_files,
        # Code generation
        'generate_code': generate_code,
        'generate_code_edit': generate_code_edit,
        # Text generation
        'write_text': write_text,
        'edit_text': edit_text,
        'write_text_from_source': write_text_from_source,
        # Dynamic tool creation
        'create_tool': create_tool,
        'list_custom_tools': list_custom_tools,
        'remove_custom_tool': remove_custom_tool,
        # Git
        'git_init': git_init,
        'git_status': git_status,
        'git_add': git_add,
        'git_commit': git_commit,
        'git_log': git_log,
        'git_diff': git_diff,
        'git_diff_staged': git_diff_staged,
        'git_branch': git_branch,
        'git_checkout': git_checkout,
        'git_list_branches': git_list_branches,
        # User interaction
        'check_in': check_in,
    })
    load_custom_tools()
    return available_functions
