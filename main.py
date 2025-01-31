import json
import re
from datetime import timedelta

import requests
from future.backports.datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time
import yaml
from pathlib import Path
import bcrypt
from functools import wraps
import os

IS_WINDOWS = os.name == 'nt'
STUNN_PUBLIC_URL = 'http://127.0.0.1:5000' if IS_WINDOWS else 'https://www.stunnvideo.com'

class SecureAuthenticator:
    def __init__(self, config_path="config/auth_config.yaml"):
        """Initialize the authenticator with configuration."""
        self.config_path = Path(config_path)
        self.max_attempts = 5
        self.lockout_time = 300  # 5 minutes
        self.session_timeout = 3600  # 1 hour
        self._load_config()

        # Initialize session state variables
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
        if 'lockout_until' not in st.session_state:
            st.session_state.lockout_until = None
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = None

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            self._create_default_config()

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def _create_default_config(self):
        """Create default configuration file."""
        self.config_path.parent.mkdir(exist_ok=True)
        default_config = {
            'users': {}
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f)

    def _hash_password(self, password: str) -> tuple:
        """Hash password using bcrypt with salt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode(), salt)
        return hashed, salt

    def add_user(self, username: str, password: str, role: str = 'user'):
        """Add a new user with hashed password."""
        if username in self.config['users']:
            raise ValueError('Username already exists')

        hashed_pw, salt = self._hash_password(password)
        self.config['users'][username] = {
            'hashed_password': hashed_pw.decode(),
            'salt': salt.decode(),
            'role': role,
            'created_at': datetime.now().isoformat()
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    def _verify_password(self, username: str, password: str) -> bool:
        """Verify password against stored hash."""
        if username not in self.config['users']:
            return False

        user = self.config['users'][username]
        stored_hash = user['hashed_password'].encode()
        return bcrypt.checkpw(password.encode(), stored_hash)

    def _check_session_timeout(self):
        """Check if the session has timed out."""
        if st.session_state.last_activity:
            elapsed = time.time() - st.session_state.last_activity
            if elapsed > self.session_timeout:
                self.logout()
                st.error('Session timed out. Please login again.')
                return False
        return True

    def _update_activity(self):
        """Update last activity timestamp."""
        st.session_state.last_activity = time.time()

    def login(self, username: str, password: str) -> bool:
        """Process login attempt with rate limiting."""
        # Check if user is locked out
        if st.session_state.lockout_until and datetime.now() < st.session_state.lockout_until:
            remaining = (st.session_state.lockout_until - datetime.now()).seconds
            st.error(f'Account is locked. Try again in {remaining} seconds.')
            return False

        # Reset lockout if expired
        if st.session_state.lockout_until and datetime.now() >= st.session_state.lockout_until:
            st.session_state.login_attempts = 0
            st.session_state.lockout_until = None

        # Verify credentials
        if self._verify_password(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = self.config['users'][username]['role']
            st.session_state.login_attempts = 0
            self._update_activity()
            return True
        else:
            st.session_state.login_attempts += 1
            if st.session_state.login_attempts >= self.max_attempts:
                st.session_state.lockout_until = datetime.now() + timedelta(seconds=self.lockout_time)
                st.error(f'Too many failed attempts. Account locked for {self.lockout_time} seconds.')
            else:
                st.error(
                    f'Invalid credentials. {self.max_attempts - st.session_state.login_attempts} attempts remaining.')
            return False

    def logout(self):
        """Log out user and clear session state."""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.last_activity = None

    def require_auth(self, func):
        """Decorator to require authentication for views."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not st.session_state.authenticated:
                st.error('Please login to access this page.')
                return
            if not self._check_session_timeout():
                return
            self._update_activity()
            return func(*args, **kwargs)

        return wrapper


# Example usage in your main.py
def initialize_auth():
    """Initialize authentication system."""
    return SecureAuthenticator()


# Initialize authentication globally
def get_auth():
    if 'auth' not in st.session_state:
        st.session_state.auth = SecureAuthenticator()
    return st.session_state.auth


def login_page():
    """Render login page."""
    st.title('Login')

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            auth = get_auth()
            auth.login(username, password)

#
# # auth = SecureAuthenticator()
# # auth.add_user('admin', 'CJLYNC01928374', role='admin')
#
# Must be the first Streamlit command
st.set_page_config(
    page_title="Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Connect to SQLite database
def fetch_analytics():
    return requests.get(
        STUNN_PUBLIC_URL + '/backdoor-api/analytics',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.environ["ANALYTICS_API_KEY"]}',
        }
    ).json()

def fetch_logs():
    if TESTING_MODE:
        return {
            'host': {'all': [{'time': '2024-11-21 23:04:46', 'text': 'Hello, this is a test\nAnother test'}, {'time': '2024-11-21 23:04:46', 'text': 'Traceback (most recent call last):\n\n\n  File "C:\\Users\\cjlyn\\PycharmProjects\\StunnVideoHost\\error_log.py", line 107, in <module>\n    1 / 0\n    ~~^~~\n\n\nZeroDivisionError: division by zero\n\n\nTraceback (most recent call last):\n\n\n  File "C:\\Users\\cjlyn\\PycharmProjects\\StunnVideoHost\\error_log.py", line 112, in <module>\n    x = undefined_variable  # This will cause a NameError\n        ^^^^^^^^^^^^^^^^^^\n\n\nNameError: name \'undefined_variable\' is not defined\n\n'}], 'print': [{'time': '2024-11-21 23:04:46', 'text': 'Hello, this is a test\nAnother test'}], 'error': [{'time': '2024-11-21 23:04:46', 'text': 'Traceback (most recent call last):\n\n\n  File "C:\\Users\\cjlyn\\PycharmProjects\\StunnVideoHost\\error_log.py", line 107, in <module>\n    1 / 0\n    ~~^~~\n\n\nZeroDivisionError: division by zero\n\n\nTraceback (most recent call last):\n\n\n  File "C:\\Users\\cjlyn\\PycharmProjects\\StunnVideoHost\\error_log.py", line 112, in <module>\n    x = undefined_variable  # This will cause a NameError\n        ^^^^^^^^^^^^^^^^^^\n\n\nNameError: name \'undefined_variable\' is not defined\n\n'}]},
            'edit_instantly': {'all': [{'time': '2024-11-21 23:04:46', 'text': 'Hello, this is a test\nAnother test'}, {'time': '2024-11-21 23:04:46', 'text': 'Traceback (most recent call last):\n\n\n  File "C:\\Users\\cjlyn\\PycharmProjects\\StunnVideoHost\\error_log.py", line 107, in <module>\n    1 / 0\n    ~~^~~\n\n\nZeroDivisionError: division by zero\n\n\nTraceback (most recent call last):\n\n\n  File "C:\\Users\\cjlyn\\PycharmProjects\\StunnVideoHost\\error_log.py", line 112, in <module>\n    x = undefined_variable  # This will cause a NameError\n        ^^^^^^^^^^^^^^^^^^\n\n\nNameError: name \'undefined_variable\' is not defined\n\n'}], 'print': [{'time': '2024-11-21 23:04:46', 'text': 'Hello, this is a test\nAnother test'}], 'error': [{'time': '2024-11-21 23:04:46', 'text': 'Traceback (most recent call last):\n\n\n  File "C:\\Users\\cjlyn\\PycharmProjects\\StunnVideoHost\\error_log.py", line 107, in <module>\n    1 / 0\n    ~~^~~\n\n\nZeroDivisionError: division by zero\n\n\nTraceback (most recent call last):\n\n\n  File "C:\\Users\\cjlyn\\PycharmProjects\\StunnVideoHost\\error_log.py", line 112, in <module>\n    x = undefined_variable  # This will cause a NameError\n        ^^^^^^^^^^^^^^^^^^\n\n\nNameError: name \'undefined_variable\' is not defined\n\n'}]}
        }
    else:
        return requests.get(
            STUNN_PUBLIC_URL + '/backdoor-api/logs',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {os.environ["ANALYTICS_API_KEY"]}',
            }
        ).json()

# Fetch analytics data
TESTING_MODE = False
def create_metric_cards(metrics, columns=4):
    cols = st.columns(columns)
    for idx, (key, value) in enumerate(metrics.items()):
        with cols[idx % columns]:
            st.metric(key.replace('_', ' ').title().replace(' Ai ', ' AI '), value)


def plot_time_series(data, title):
    """Plot time series data with error handling and proper serialization"""
    if not data:
        st.info(f"No data available for {title}")
        return

    try:
        # Convert data to DataFrame
        df = pd.DataFrame(list(data.items()), columns=['Date', 'Value'])
        df['Date'] = pd.to_datetime(df['Date'])

        # Create figure with go instead of px
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Value'],
                mode='lines+markers',
                name=title
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_dark",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Use streamlit's native plotting
        st.plotly_chart(fig, use_container_width=True, theme=None, key='plotly'+title)

    except Exception as e:
        st.error(f"Error plotting {title}: {str(e)}")
        st.write("Data:", data)  # Debug info


def plot_time_series_alt(data, title):
    """Alternative plotting using Altair if Plotly fails"""
    if not data:
        st.info(f"No data available for {title}")
        return

    try:
        # Convert to DataFrame
        df = pd.DataFrame(list(data.items()), columns=['Date', 'Value'])
        df['Date'] = pd.to_datetime(df['Date'])

        # Create chart using streamlit's native line chart
        st.subheader(title)
        st.line_chart(
            data=df.set_index('Date'),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error plotting {title}: {str(e)}")
        st.write("Raw data:", data)

def plot_table_data(data, title, key_prefix):
    if data:
        for user in data:
            if 'profile_picture' in user:
                user['profile_picture'] = make_full_and_protect(user['profile_picture'])
                
        df = pd.DataFrame(data)

        # Select columns to display
        exclude_cols = ['_password', 'no_of_minutes_left', 'no_of_minutes_left_pivot', 'payAsYouGoSetup',
                        'setupIntent',
                        'stripeCustomerId',
                        'stripeCheckoutSessionId',
                        'stripeSubscriptionId', 'user_data']  # Removed profile_picture from exclude_cols
        cols_to_display = [col for col in df.columns if col not in exclude_cols]

        # Handle profile pictures for users and feedback sections
        if title.lower() in ['users', 'feedbacks', 'users usage'] and 'profile_picture' in df.columns:
            # Just pass the URL directly - no need for st.image()
            df['Profile'] = df['profile_picture']

            # Reorder columns to show profile picture first
            cols_to_display.remove('profile_picture')
            cols_to_display = ['Profile'] + cols_to_display

        # Add filters for boolean and integer columns
        bool_cols = df.select_dtypes(include=['bool', 'int64']).columns
        bool_cols = [col for col in bool_cols if col not in exclude_cols]

        if len(bool_cols) > 0:
            st.write("Exclude Filters:")
            filter_cols = st.columns(len(bool_cols))
            filters = {}
            for idx, col in enumerate(bool_cols):
                with filter_cols[idx]:
                    filters[col] = st.multiselect(
                        f"Exclude {col.replace('_', ' ').title().replace(' Ai ', ' AI ')}",
                        options=[0, 1],
                        default=[],
                        key=f"{key_prefix}_{col}"
                    )

            # Apply exclusion filters
            for col, values in filters.items():
                if values:
                    df = df[~df[col].isin(values)]

        # Configure columns with explicit pixel widths
        USAGE_KEYS = ['chars_generated', 'exports', 'ai_art_generated']
        column_config = {
            "Profile": st.column_config.ImageColumn(
                "Profile",
                width=32,  # Set explicit pixel width
            ),
            "id": st.column_config.Column(
                "ID",
                width=27,  # Set explicit pixel width
            ),
            "_id": st.column_config.Column(
                "ID",
                width=5,  # Set explicit pixel width
            ),
            **{
                f"daily_{key}": st.column_config.LineChartColumn(
                    f"Daily {key.replace('_', ' ').title().replace(' Ai ', ' AI ')}",
                    width=200,  # Set explicit pixel width
                )
                for key in USAGE_KEYS
            }
        }

        # convert daily_chars_generated in dict format ex: {"2024-12-20":414,"2024-12-21":1305} to LineChartColumn
        for key in USAGE_KEYS:
            if f'daily_{key}' in df.columns:
                df[f'daily_{key}'] = df[f'daily_{key}'].apply(
                    lambda x: [chars for date, chars in x.items()]
                )



        # Display table with pagination
        page_size = 100
        total_pages = len(df) // page_size + (1 if len(df) % page_size != 0 else 0)
        if total_pages > 0:
            page = st.number_input(f'Page ({total_pages} total)', min_value=1, max_value=total_pages, value=1,
                                   key=f"{key_prefix}_page")
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            st.dataframe(
                df[cols_to_display].iloc[start_idx:end_idx],
                use_container_width=True,
                column_config=column_config,
                hide_index=True
            )
        else:
            st.dataframe(
                df[cols_to_display],
                use_container_width=True,
                column_config=column_config,
                hide_index=True
            )
    else:
        st.info(f"No data available for {title}")

# Add this to your CSS
def inject_custom_css():
    st.markdown("""
        <style>
        /* Log table styles */
        .log-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 8px;
        }
        .log-row {
            background: #1e1e1e;
            margin-bottom: 8px;
        }
        .log-timestamp {
            color: #666;
            font-size: 0.85em;
            padding: 12px;
            width: 180px;
            vertical-align: top;
        }
        .log-content {
            padding: 12px;
            font-family: ui-monospace, "Roboto Mono", monospace;
            white-space: pre-wrap;
        }
        .log-content.error {
            color: #ff4b4b;
        }
        .log-content.print {
            color: #ffffff;
        }
        /* Custom styles for streamlit elements */
        div[data-testid="stVerticalBlock"] > div:has(div.log-table) {
            gap: 0rem;
        }
        </style>
    """, unsafe_allow_html=True)


import html


def format_log_text(text):
    """Format log text for HTML display"""
    # Escape special characters for HTML
    escaped_text = html.escape(text)
    # Replace newlines with <br> tags
    formatted_text = escaped_text.replace('\n', '<br>')
    return formatted_text


import html


def create_logs_component(logs_data):
    """Create logs component for display"""
    css = """
    <style>
        .log-entry {
            background-color: #1e1e1e;
            border-radius: 4px;
            padding: 12px 16px;
            margin: 8px 0;
        }
        .log-entry .timestamp {
            color: #888;
            font-size: 0.8em;
            font-family: 'Courier New', monospace;
            margin-bottom: 4px;
        }
        .log-entry .content {
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.5;
        }
        .log-entry .content.error {
            color: #ff4b4b;
        }
        .log-entry .content.print {
            color: #ffffff;
        }
        .stButton button {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #444;
        }
        .stTextInput input {
            background-color: #2b2b2b;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def display_logs_table(logs_data, page_size=50, name="System Logs"):
    """Display logs with efficient pagination and filtering"""

    create_logs_component(logs_data)
    st.markdown(f"### {name}")

    # Filters
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        log_type = st.selectbox("Log Type", ['all', 'error', 'print'], key=name+'box')
    with col2:
        date_filter = st.date_input("Date Filter", value=None, key=name+'date_b_box')
    with col3:
        search_term = st.text_input("Search logs", key=name+'search-box').lower()

    # Filter logs
    filtered_logs = logs_data[log_type]

    if date_filter:
        filtered_logs = [
            log for log in filtered_logs
            if datetime.strptime(log['time'].split()[0], '%Y-%m-%d').date() == date_filter
        ]

    if search_term:
        filtered_logs = [
            log for log in filtered_logs
            if search_term in log['text'].lower()
        ]

    # Reverse the order - most recent first
    filtered_logs = list(reversed(filtered_logs))

    # Pagination
    total_logs = len(filtered_logs)
    total_pages = (total_logs + page_size - 1) // page_size

    if total_pages > 0:
        col1, col2 = st.columns([3, 2])
        with col1:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key=name+'2')
        with col2:
            st.markdown(f"**Total entries:** {total_logs}")

        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_logs)

        # Display each log entry separately
        for log in filtered_logs[start_idx:end_idx]:
            # Format timestamp
            dt = datetime.strptime(log['time'], '%Y-%m-%d %H:%M:%S')
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')

            # Determine log type
            is_error = 'Traceback' in log['text']

            # Create columns for the log entry
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"""
                    <div style="color: #888; font-size: 0.8em;
                       font-family: monospace; padding: 8px;">{formatted_time}
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                color = "#ff4b4b" if is_error else "#ffffff"
                st.markdown(f"""
                    <div style="color: {color}; font-family: monospace;
                       white-space: pre-wrap; padding: 8px;">{format_log_text(log['text'])}
                    </div>""", unsafe_allow_html=True)

            # Add a subtle divider
            st.markdown("<hr style='border: none; border-top: 1px solid #333; margin: 4px 0;'>",
                        unsafe_allow_html=True)
    else:
        st.info("No logs found matching the criteria")


def create_usage_chart(usage_stats):
    """Creates a small horizontal chart for project usage statistics with dark mode theme."""
    dates = list(usage_stats.keys())
    values = list(usage_stats.values())

    fig = go.Figure(data=go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        line=dict(width=2, color='#60A5FA'),  # Bright blue that works well in dark mode
        marker=dict(
            size=4,
            color='#60A5FA',
            line=dict(
                color='#93C5FD',
                width=1
            )
        )
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=50,
        showlegend=False,
        plot_bgcolor='#0E1117',  # Dark background
        paper_bgcolor='#0E1117',  # Dark background for the entire plot
        font=dict(
            color='#E5E7EB'  # Light gray text for better contrast
        )
    )

    # Update axes to match dark theme
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        showline=False,
        zeroline=False
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        showline=False,
        zeroline=False
    )

    return fig


def format_date(date_str: str) -> str:
    """Formats date string to a more readable format."""
    date = datetime.fromisoformat(date_str)
    return date.strftime("%b %d, %Y %H:%M")


def make_full_and_protect(url):
    # TODO: USED IN EDIT_BACKEND | MAKE SURE IT IS NOT USED IN A DANGEROUS WAY
    # TODO: USED FOR DOWNLOADING AUDIO | MAKE SURE IT IS NOT USED IN A DANGEROUS WAY
    # TODO: IMPLEMENT SAFE GUARDS AND CHECKS, EVERY CALL FROM STUNN MUST GIVE A SECRET KEY TO ACCESS BACKED
    url = re.sub(r'\n', '', url)
    if url.startswith('/'):
        url = STUNN_PUBLIC_URL + url
    else:
        assert url.startswith('https://')
    return url.strip()


def render_project_card(project, idx):
    """Renders a single project card with all required information."""
    st.markdown("""
    <style>
    [data-testid="stExpander"] details {
        border-style: none;
    }
    .chat-box {
        background-color: #08080f;
        padding: 2px;
        border-radius: 5px;
        margin-bottom: 10px;
        padding-left: 14px;
    }
    .chat-message {
        margin: 5px 0;
        background-color: none;
    }
    .chat-user {
        font-weight: bold;
        color: #007bff;
    }
    .chat-ai {
        font-weight: bold;
        color: #28a745;
    }
    </style>
        """, unsafe_allow_html=True)

    with st.container(border=True):
        # col1 = st.columns([20])
        # with col1:
        col_thumb, title_col, date1, date2 = st.columns([1, 7, 2, 2])
        with col_thumb:
            st.image(make_full_and_protect(project['thumbnail'].replace('..', '')), width=60, use_container_width=False)
        with title_col:
            title = project.get('title', 'Untitled Project')
            if len(title) > 40:
                title = title[:37] + '...'
            liked_emoji = '👍' if project.get('liked', False) else '👎' if project.get('disliked', False) else ''
            st.markdown(f"#### {title} {liked_emoji} \n\n <i>{project.get('prompt', 'No Prompt Specified')}</i>", unsafe_allow_html=True)
        with date1:
            st.caption(f"📅 {format_date(project['date_created'])}")
        with date2:
            st.caption(f"🔄 {format_date(project['date_last_modified'])}")

        # Metadata row
        meta_cols = st.columns([0.6, 0.9, 0.4, 0.5, 0.4, 1, 0.6, 0.4, 0.8, 0.2])
        with meta_cols[0]:
            st.caption(f"⏱️ {project['duration']:.2f}s")
        with meta_cols[1]:
            hours_spent = len(project.get('usage_stats', {}))
            st.caption(f"📈 < {hours_spent:.2f}h")
        with meta_cols[2]:
            st.caption(f"📊 {project['total_contents']}")
        with meta_cols[3]:
            st.caption(f"⬇️ {project['total_downloads']}")
        with meta_cols[4]:
            if 'owner_profile_picture' in project and project['owner_profile_picture'] is not None:
                st.image(make_full_and_protect(project['owner_profile_picture'].replace('..', '')), width=23)
        with meta_cols[5]:
            privacy_color = {
                'private': '🔴',
                'anyone_can_view': '🟢',
                'anyone_can_edit': '🔵'
            }.get(project['privacy'], '⚪')
            st.caption(f"{privacy_color} {project['owner_name'] if project['owner_name'] is not None else 'Signed Out'}")
        with meta_cols[6]:
            st.caption(f"@{project['owner_username'] if project['owner_username'] is not None else ''}")
        with meta_cols[7]:
            st.caption(f"🏆 {sum(project.get('usage_stats', {}).values())}")
        with meta_cols[8]:
            st.caption(f"{project.get('storage_used_mb', 0.0):.2f}MB")
        with meta_cols[9]:
            # link to take us to STUNN_PUBLIC_URL + /p/{project['id']}
            st.markdown(f"[🔗]({STUNN_PUBLIC_URL}/p/{project['id']})")

        # Social media description with expansion
        text = project.get('text', "No description available.\n") + project.get('social_media_description', "")
        with st.expander(text[:237] + '...' if len(text) > 240 else text, expanded=False):
            st.write(text[237:])
            chat_history = project.get('chat_history', [])[::-1]
            # mini chat box
            if chat_history:
                st.markdown("##### Chat History")
                for history in chat_history:
                    source_class = "chat-user" if history['source'] == "user" else "chat-ai"
                    st.markdown(
                       f"""
                        <div class="chat-box">
                            <div class="chat-message">
                                <span class="{source_class}">{history['source'].capitalize().replace('Ai', 'Stunn')}</span>
                                {history['text']}
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )


        # Usage statistics chart
        if project.get('usage_stats'):
            st.plotly_chart(create_usage_chart(project['usage_stats']), use_container_width=True, key=f'{idx}')

        # Chat history p


def view_projects_tab(user_data):
    """Main function to render the projects tab."""
    st.title("Projects")

    # Filters and sorting options
    col1, col2, col3 = st.columns(3)

    with col1:
        search = st.text_input("🔍 Search projects")

    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Recent", "Oldest", "Most Used", "Title", "Duration"]
        )

    with col3:
        privacy_filter = st.multiselect(
            "Privacy",
            ["Private", "Anyone Can View", "Anyone Can Edit"]
        )

    # Time range filter for usage stats
    # time_range = st.slider(
    #     "Usage time range",
    #     min_value=datetime(2024, 1, 1),
    #     max_value=datetime(2024, 12, 31),
    #     value=(datetime(2024, 1, 1), datetime(2024, 12, 31))
    # )

    # Fetch and process projects
    projects = user_data['projects']

    # Apply filters
    filtered_projects = []
    for project in projects:
        assert isinstance(project, dict)
        search_q = (project.get('text', '').lower() +
                    project.get('social_media_description', '').lower() +
                    (project.get('owner_name', '') or ' ').lower() +
                    (project.get('owner_username', '') or ' ').lower() +
                    (project.get('owner_email', '') or ' ').lower() +
                    project.get('title', '').lower() +
                    project.get('id', '').lower() +
                    project.get('privacy', '').lower() +
                    str(project.get('duration', '')).lower() +
                    project.get('date_created', '').lower() +
                    project.get('date_last_modified', '').lower())
        if search and search.lower() not in search_q:
            continue

        if privacy_filter and project['privacy'].title().replace('_', ' ') not in privacy_filter:
            continue

        if len(project) < 8:
            continue

        filtered_projects.append(project)

    # Sort projects
    if sort_by == "Recent":
        filtered_projects.sort(key=lambda x: x['date_last_modified'], reverse=True)
    elif sort_by == "Oldest":
        filtered_projects.sort(key=lambda x: x['date_created'])
    elif sort_by == "Most Used":
        filtered_projects.sort(key=lambda x: sum(x.get('usage_stats', {}).values()), reverse=True)
    elif sort_by == "Title":
        filtered_projects.sort(key=lambda x: x.get('text', '').lower())
    elif sort_by == "Duration":
        filtered_projects.sort(key=lambda x: x['duration'], reverse=True)

    # Pagination
    projects_per_page = 50
    total_pages = (len(filtered_projects) + projects_per_page - 1) // projects_per_page

    if total_pages > 0:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * projects_per_page
        end_idx = start_idx + projects_per_page

        # Display projects for current page
        for idx, project in enumerate(filtered_projects[start_idx:end_idx], start=start_idx):
            render_project_card(project, idx)
    else:
        st.info("No projects found matching the current filters.")

def main():
    auth = initialize_auth()
    if not st.session_state.authenticated:
        login_page()
    else:
        @auth.require_auth
        def authenticated_main():
            # st.set_page_config(
            #     page_title="Analytics Dashboard",
            #     layout="wide",
            #     initial_sidebar_state="collapsed"
            # )

            # Inject custom CSS
            inject_custom_css()

            st.title("Stunn Analytics Dashboard")
            data = fetch_analytics()
            logs_data = fetch_logs()

            # Create tabs
            metrics_tab, time_series_tab, tables_tab, host_logs_tab, edit_instantly_logs_tab, projects_tab = st.tabs([
                "Key Metrics",
                "Time Series",
                "Detailed Data",
                "Host Logs",
                "EditInstantly Logs",
                "Projects"
            ])
            with projects_tab:
                view_projects_tab(data)

            with metrics_tab:
                st.subheader("Overall Metrics")
                overall_metrics = {
                    'total_users': data['total_users'],
                    'total_visitors': data['total_visitors'],
                    'total_visits': data['total_visits'],
                    'total_feedbacks': data['total_feedbacks'],
                    'total_projects': data['total_projects'],
                    'total_storage': f"{data['total_storage']:.2f}GB",
                    'total_unsubscribes': data['total_unsubscribes']
                }
                create_metric_cards(overall_metrics)

                st.subheader("Usage Metrics")
                HISTORY_TYPES = [
                    'ai_art_generated',
                    'chars_generated',
                    'exports'
                ]
                meta_metrics = {
                    'total_active_visitors': data['total_active_visitors'],
                }
                for key in HISTORY_TYPES:
                    meta_metrics[f'total_{key}'] = data[f'total_{key}']

                create_metric_cards(meta_metrics)
                st.subheader("Current Period Metrics")
                period_metrics = {
                    'total_users_this_day': data['total_users_this_day'],
                    'total_visits_this_day': data['total_visits_this_day'],
                    'total_users_this_week': data['total_users_this_week'],
                    'total_visits_this_week': data['total_visits_this_week'],
                    'total_users_this_month': data['total_users_this_month'],
                    'total_visits_this_month': data['total_visits_this_month'],
                    'total_unsubscribes_this_day': data['total_unsubscribes_this_day']
                }
                create_metric_cards(period_metrics)

                st.subheader("Current Period Usage Metrics")
                meta_metrics = {}
                for key in HISTORY_TYPES:
                    meta_metrics[f'{key}_today'] = data[f'{key}_today']
                create_metric_cards(period_metrics)

            with time_series_tab:
                col1, col2 = st.columns(2)

                PLOTS = [
                    'daily_users', 'daily_visitors', 'daily_projects',
                    'daily_ai_art_generated', 'daily_chars_generated', 'daily_exports'
                ]
                for key in PLOTS:
                    with col1:
                        title = key.replace('_', ' ').title().replace(' Ai ', ' AI ')
                        st.subheader(title)
                        plot_time_series(data[key], f"{title} Trend")

            with tables_tab:
                # Main data tables
                for section, title in [
                    ('users', 'Users'),
                    ('users_usage', 'Users Usage'),
                    ('feedbacks', 'Feedbacks'),
                    ('site_requests', 'Site Requests'),
                    ('unsubscribes', 'Unsubscribes'),
                    ('projects', 'Projects')
                ]:
                    with st.expander(f"{title} Data", expanded=section in ['users', 'feedbacks', 'users_usage']):
                        plot_table_data(data[section], title, section)

                        # Special visualizations for site requests
                        if section == 'site_requests' and data[section]:
                            for a in data[section]:
                                if 'user_agent' in a['lang']:
                                    a['lang'] = json.loads(a['lang'])['lang']

                            df = pd.DataFrame(data[section])
                            col1, col2 = st.columns(2)

                            with col1:
                                url_counts = df['url'].value_counts()
                                fig = px.pie(values=url_counts.values, names=url_counts.index, title='URL Distribution')
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                lang_counts = df['lang'].value_counts()
                                fig = px.bar(x=lang_counts.index, y=lang_counts.values, title='Language Distribution')
                                st.plotly_chart(fig, use_container_width=True)

                            col1, col2 = st.columns(2)
                            with col1:
                                url_counts = [a for a in data['referrer_distribution'].items()]
                                fig = px.pie(values=[a[1] for a in url_counts], names=[a[0] for a in url_counts], title='Referrer Distribution')
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                url_counts = [a for a in data['referrer_distribution_this_month'].items()]
                                fig = px.pie(values=[a[1] for a in url_counts], names=[a[0] for a in url_counts], title='Referrer Distribution This Month')
                                st.plotly_chart(fig, use_container_width=True)

                            col1, col2 = st.columns(2)
                            with col1:
                                url_counts = [a for a in data['ref_distribution'].items()]
                                fig = px.pie(values=[a[1] for a in url_counts], names=[a[0] for a in url_counts], title='Ref Distribution')
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                url_counts = [a for a in data['ref_distribution_this_month'].items()]
                                fig = px.pie(values=[a[1] for a in url_counts], names=[a[0] for a in url_counts], title='Ref Distribution This Month')
                                st.plotly_chart(fig, use_container_width=True)

            with host_logs_tab:
                display_logs_table(logs_data['host'], name="Host Logs")

            with edit_instantly_logs_tab:
                display_logs_table(logs_data['edit_instantly'], name="EditInstantly Logs")

        authenticated_main()


try:
    auth = SecureAuthenticator()
    auth.add_user('admin', 'CJLYNC01928374', role='admin')
    print('Adding User Success')
except Exception as e:
    print('Adding User Failed')

if __name__ == "__main__":
    main()
