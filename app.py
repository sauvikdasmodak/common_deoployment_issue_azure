import json
import os
import re
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from groq import AuthenticationError as GroqAuthenticationError
from groq import Groq, RateLimitError as GroqRateLimitError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import OpenAI, RateLimitError as OpenAIRateLimitError

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Azure Deployment Issue Analyzer",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] { background: #0a0a0a; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #aaa !important; }

.issue-critical {
    background: #2d1b1b;
    border-left: 4px solid #e53e3e;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}
.issue-warning {
    background: #2d2410;
    border-left: 4px solid #ed8936;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}
.issue-info {
    background: #132233;
    border-left: 4px solid #4299e1;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}
.issue-success {
    background: #1a2d1a;
    border-left: 4px solid #48bb78;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}
.badge-critical { background:#e53e3e; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-warning  { background:#ed8936; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-info     { background:#4299e1; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-low      { background:#48bb78; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }

.metric-box {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-number { font-size: 2rem; font-weight: 700; }
.metric-label  { font-size: 0.85rem; color: #aaa; margin-top: 4px; }

code { background: #1e1e2e; padding: 2px 6px; border-radius: 3px; font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
AZURE_SERVICES = [
    "All Azure Services",
    "Azure App Service",
    "Azure Kubernetes Service (AKS)",
    "Azure Container Instances",
    "Azure Functions",
    "Azure Virtual Machines",
    "Azure SQL Database",
    "Azure Cosmos DB",
    "Azure Storage",
    "Azure API Management",
    "Azure DevOps Pipelines",
    "Azure Resource Manager (ARM/Bicep)",
    "Azure Active Directory / Entra ID",
    "Azure Networking (VNet/NSG/LB)",
    "Azure Event Hub / Service Bus",
    "Azure Logic Apps",
    "Azure Monitor / Log Analytics",
]

LLM_PROVIDERS = {
    "Groq": {
        "env_var": "GROQ_API_KEY",
        "placeholder": "gsk_...",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "llama-3.1-8b-instant",
        ],
        "default_model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    },
    "OpenAI": {
        "env_var": "OPENAI_API_KEY",
        "placeholder": "sk-...",
        "models": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "default_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    },
}

SYSTEM_PROMPT = """You are an expert Azure cloud engineer and DevOps specialist.
Your task is to analyze Azure deployment error logs and identify all deployment issues.

For each issue found, provide a structured analysis in valid JSON format with the following schema:
{
  "summary": "One-line overall summary of log health",
  "total_issues": <integer>,
  "issues": [
    {
      "id": <integer starting from 1>,
      "title": "Short descriptive title",
      "severity": "critical|warning|info|resolved",
      "azure_service": "Affected Azure service name",
      "error_code": "Azure error code if present (e.g. ResourceNotFound, QuotaExceeded) or null",
      "description": "Detailed description of the issue",
      "root_cause": "Likely root cause",
      "fix": "Step-by-step recommended fix",
      "log_snippet": "Relevant log lines that indicate this issue (max 3 lines)",
      "tags": ["list", "of", "relevant", "tags"]
    }
  ],
  "recommendations": [
    "Overall recommendation 1",
    "Overall recommendation 2"
  ],
  "health_score": <integer 0-100 representing overall deployment health>
}

Rules:
- Only return valid JSON, no extra text before or after.
- Be thorough — identify ALL issues including auth errors, quota limits, network problems, misconfiguration, missing resources, permission errors, timeouts, and service-specific issues.
- Severity guide: critical=deployment failed/blocked, warning=degraded/risky, info=noteworthy/non-blocking, resolved=error that was auto-recovered.
- If the logs are clean with no issues, return total_issues: 0 and an empty issues array with health_score: 100.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_default_provider() -> str:
    if os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY"):
        return "OpenAI"
    return "Groq"


def get_client(provider: str, api_key: str):
    if provider == "OpenAI":
        return OpenAI(api_key=api_key)
    return Groq(api_key=api_key)


def analyze_logs(client, model: str, log_text: str, service_filter: str) -> dict:
    service_hint = ""
    if service_filter and service_filter != "All Azure Services":
        service_hint = f"\nFocus especially on issues related to: {service_filter}."

    user_message = f"""Analyze the following Azure deployment logs and return a JSON report of all issues found.{service_hint}

=== LOGS START ===
{log_text}
=== LOGS END ===
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if model wraps output
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def severity_badge(severity: str) -> str:
    mapping = {
        "critical": '<span class="badge-critical">CRITICAL</span>',
        "warning":  '<span class="badge-warning">WARNING</span>',
        "info":     '<span class="badge-info">INFO</span>',
        "resolved": '<span class="badge-low">RESOLVED</span>',
    }
    return mapping.get(severity.lower(), f'<span class="badge-info">{severity.upper()}</span>')


def issue_card_class(severity: str) -> str:
    return {
        "critical": "issue-critical",
        "warning":  "issue-warning",
        "info":     "issue-info",
        "resolved": "issue-success",
    }.get(severity.lower(), "issue-info")


def health_color(score: int) -> str:
    if score >= 80:
        return "#48bb78"
    if score >= 50:
        return "#ed8936"
    return "#e53e3e"


def render_issue(issue: dict):
    css_class = issue_card_class(issue.get("severity", "info"))
    badge = severity_badge(issue.get("severity", "info"))
    title = issue.get("title", "Unknown Issue")
    service = issue.get("azure_service", "—")
    error_code = issue.get("error_code")
    description = issue.get("description", "")
    root_cause = issue.get("root_cause", "")
    fix = issue.get("fix", "")
    snippet = issue.get("log_snippet", "")
    tags = issue.get("tags", [])

    error_code_html = f'<code>{error_code}</code>' if error_code else ""
    tags_html = " ".join(f'<code>{t}</code>' for t in tags)

    st.markdown(f"""
<div class="{css_class}">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    {badge}
    <strong style="font-size:1rem;">{title}</strong>
    <span style="margin-left:auto;font-size:0.8rem;color:#aaa;">🔷 {service}</span>
  </div>
  {"<p style='margin:0 0 4px;font-size:0.82rem;color:#aaa;'>Error Code: " + error_code_html + "</p>" if error_code else ""}
  <p style="margin:6px 0;font-size:0.9rem;">{description}</p>
</div>
""", unsafe_allow_html=True)

    with st.expander("Root Cause & Fix"):
        st.markdown(f"**Root Cause:** {root_cause}")
        st.markdown(f"**Recommended Fix:**\n{fix}")
        if snippet:
            st.code(snippet, language="text")
        if tags:
            st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☁️ Azure Log Analyzer")
    st.markdown("---")

    st.markdown("### 🔑 LLM Configuration")

    provider_name = st.selectbox(
        "Provider",
        list(LLM_PROVIDERS.keys()),
        index=list(LLM_PROVIDERS.keys()).index(get_default_provider()),
    )
    provider_config = LLM_PROVIDERS[provider_name]
    api_env_var = provider_config["env_var"]
    available_models = provider_config["models"]
    default_model = provider_config["default_model"]
    default_model_index = available_models.index(default_model) if default_model in available_models else 0

    api_key = st.text_input(
        f"{provider_name} API Key",
        value=os.getenv(api_env_var, ""),
        type="password",
        placeholder=provider_config["placeholder"],
    )

    model_name = st.selectbox(
        "Model",
        available_models,
        index=default_model_index,
    )

    st.markdown("---")
    st.markdown("### 🔍 Filter")
    service_filter = st.selectbox("Azure Service", AZURE_SERVICES)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Paste or upload Azure deployment logs. "
        "The LLM analyzes them and categorizes every deployment issue with root causes and fixes."
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("☁️ Azure Deployment Issue Analyzer")
st.markdown("Analyze Azure deployment error logs using an LLM to surface, categorize, and fix deployment issues.")

tab_input, tab_results, tab_raw = st.tabs(["📋 Log Input", "🔍 Analysis Results", "🗂️ Raw JSON"])

# ── Tab 1: Input ──────────────────────────────────────────────────────────────
with tab_input:
    col_left, col_right = st.columns([3, 1])

    with col_left:
        log_input_method = st.radio(
            "Input method",
            ["Paste logs", "Upload log file", "Use sample logs"],
            horizontal=True,
        )

    log_text = ""

    if log_input_method == "Paste logs":
        log_text = st.text_area(
            "Paste your Azure deployment logs here",
            height=400,
            placeholder="Paste error logs, ARM deployment outputs, AKS events, pipeline logs, etc...",
        )

    elif log_input_method == "Upload log file":
        uploaded = st.file_uploader(
            "Upload a .log, .txt, or .json log file",
            type=["log", "txt", "json"],
        )
        if uploaded:
            log_text = uploaded.read().decode("utf-8", errors="replace")
            st.code(log_text[:3000] + ("\n...[truncated for preview]" if len(log_text) > 3000 else ""), language="text")

    elif log_input_method == "Use sample logs":
        sample_choice = st.selectbox("Choose a sample scenario", [
            "AKS Pod CrashLoopBackOff + OOMKilled",
            "ARM Template Deployment Failures",
            "Azure App Service – 503 & Quota Exceeded",
            "Azure DevOps Pipeline – Permission & Timeout Errors",
            "Azure SQL – Connection & Firewall Issues",
        ])
        log_text = SAMPLE_LOGS.get(sample_choice, "")
        if log_text:
            st.code(log_text, language="text")

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Logs", type="primary", use_container_width=True)

    if analyze_btn:
        if not api_key:
            st.error(f"Please enter your {provider_name} API key in the **LLM Configuration** section.")
        elif not log_text.strip():
            st.error("Please provide log content before analyzing.")
        else:
            with st.spinner(f"Sending logs to {provider_name} for analysis…"):
                try:
                    client = get_client(provider_name, api_key=api_key)
                    result = analyze_logs(client, model_name, log_text, service_filter)
                    st.session_state["analysis"] = result
                    st.session_state["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("Analysis complete! Switch to the **Analysis Results** tab.")
                except json.JSONDecodeError as e:
                    st.error(f"LLM returned malformed JSON. Try again or switch to a more capable model.\n\nDetail: {e}")
                except (GroqAuthenticationError, OpenAIAuthenticationError):
                    st.error(f"Invalid {provider_name} API key. Check your {api_env_var} value and try again.")
                except (GroqRateLimitError, OpenAIRateLimitError):
                    st.error(f"{provider_name} rate limit reached. Wait a moment and retry.")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

# ── Tab 2: Results ────────────────────────────────────────────────────────────
with tab_results:
    if "analysis" not in st.session_state:
        st.info("No analysis yet. Paste logs in the **Log Input** tab and click **Analyze Logs**.")
    else:
        result = st.session_state["analysis"]
        issues = result.get("issues", [])
        total = result.get("total_issues", len(issues))
        score = result.get("health_score", 0)
        summary = result.get("summary", "")
        recommendations = result.get("recommendations", [])
        analyzed_at = st.session_state.get("analyzed_at", "")

        # Severity counts
        counts = {"critical": 0, "warning": 0, "info": 0, "resolved": 0}
        for iss in issues:
            sev = iss.get("severity", "info").lower()
            counts[sev] = counts.get(sev, 0) + 1

        # ── Metrics row ──
        m1, m2, m3, m4, m5 = st.columns(5)
        score_color = health_color(score)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-number" style="color:{score_color}">{score}</div>
                <div class="metric-label">Health Score</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-number" style="color:#e53e3e">{counts['critical']}</div>
                <div class="metric-label">Critical</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-number" style="color:#ed8936">{counts['warning']}</div>
                <div class="metric-label">Warnings</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-number" style="color:#4299e1">{counts['info']}</div>
                <div class="metric-label">Info</div></div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-number" style="color:#48bb78">{counts['resolved']}</div>
                <div class="metric-label">Resolved</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        if summary:
            st.markdown(f"**Summary:** {summary}")
        if analyzed_at:
            st.caption(f"Analyzed at: {analyzed_at}")

        # ── Severity filter ──
        sev_filter = st.multiselect(
            "Filter by severity",
            ["critical", "warning", "info", "resolved"],
            default=["critical", "warning", "info", "resolved"],
        )

        filtered_issues = [i for i in issues if i.get("severity", "info").lower() in sev_filter]

        if not filtered_issues:
            st.success("No issues found matching the selected severity filters.")
        else:
            st.markdown(f"### Showing {len(filtered_issues)} issue(s)")
            for issue in filtered_issues:
                render_issue(issue)

        # ── Recommendations ──
        if recommendations:
            st.markdown("---")
            st.markdown("### 💡 Overall Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

        # ── Export ──
        st.markdown("---")
        export_json = json.dumps(result, indent=2)
        st.download_button(
            "⬇️ Export Analysis as JSON",
            data=export_json,
            file_name=f"azure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

# ── Tab 3: Raw JSON ───────────────────────────────────────────────────────────
with tab_raw:
    if "analysis" not in st.session_state:
        st.info("No analysis yet.")
    else:
        st.json(st.session_state["analysis"])

# ── Sample Logs (defined after UI so tabs render first) ───────────────────────
SAMPLE_LOGS = {
    "AKS Pod CrashLoopBackOff + OOMKilled": """\
2026-06-28T08:12:01Z  WARNING  pod/api-deployment-7d9f6b-xk2pq  Back-off restarting failed container
2026-06-28T08:12:01Z  ERROR    pod/api-deployment-7d9f6b-xk2pq  CrashLoopBackOff: back-off 5m0s restarting failed container api in pod api-deployment-7d9f6b-xk2pq_default
2026-06-28T08:11:45Z  ERROR    pod/api-deployment-7d9f6b-xk2pq  OOMKilled: container api exceeded memory limit of 256Mi
2026-06-28T08:10:30Z  ERROR    node/aks-nodepool1-38291001-vmss000002  node condition MemoryPressure=True
2026-06-28T08:09:00Z  WARNING  deployment/api-deployment  Readiness probe failed: HTTP probe failed with statuscode: 503
2026-06-28T08:08:55Z  INFO     HorizontalPodAutoscaler/api-hpa  unable to scale: current replicas=5 desired replicas=10 max replicas=5
2026-06-28T08:05:00Z  ERROR    imagepullbackoff  Failed to pull image "myregistry.azurecr.io/api:v2.1.0": rpc error: code=Unknown desc=failed to pull and unpack image: failed to resolve reference "myregistry.azurecr.io/api:v2.1.0": unexpected status code 401 Unauthorized
""",
    "ARM Template Deployment Failures": """\
2026-06-28T10:00:12Z  ERROR  DeploymentFailed  Resource Microsoft.Web/serverfarms/myAppServicePlan failed with message: 'The subscription is not registered to use namespace Microsoft.Web. See https://aka.ms/rps-not-found for how to register subscriptions.'  ErrorCode: MissingSubscriptionRegistration
2026-06-28T10:00:15Z  ERROR  DeploymentFailed  Resource Microsoft.Sql/servers/mysqlserver failed: The server name 'mysqlserver' is already taken. Please try a different server name.  ErrorCode: ServerNameAlreadyExists
2026-06-28T10:00:20Z  ERROR  Authorization  The client 'sp-deploy@contoso.com' with object id 'xxxxxxxx' does not have authorization to perform action 'Microsoft.KeyVault/vaults/secrets/write' over scope '/subscriptions/.../resourceGroups/rg-prod/providers/Microsoft.KeyVault/vaults/kv-prod'.  ErrorCode: AuthorizationFailed
2026-06-28T10:00:25Z  ERROR  QuotaExceeded  Operation results in exceeding quota limits of Core. Maximum allowed: 20, Current in use: 20, Additional requested: 4.  ErrorCode: QuotaExceeded
2026-06-28T10:00:30Z  ERROR  ResourceNotFound  The Resource 'Microsoft.Network/virtualNetworks/vnet-prod' under resource group 'rg-networking' was not found.  ErrorCode: ResourceNotFound
""",
    "Azure App Service – 503 & Quota Exceeded": """\
2026-06-28T09:00:00Z  ERROR  Microsoft.Web  Scaling failed for App Service Plan myAppServicePlan. Current tier: Standard S1. Reason: Quota exceeded for subscription.
2026-06-28T09:01:30Z  ERROR  IIS  Service Unavailable 503 - Application Pool 'myapp' is stopped
2026-06-28T09:02:00Z  WARNING  Microsoft.Web  Deployment slot swap failed: Target slot 'production' is not ready. Health check endpoint /health returned 503.
2026-06-28T09:03:15Z  ERROR  Microsoft.Web  Failed to deploy ZIP package. Reason: Disk quota exceeded on host. Available: 0 MB, Required: 150 MB.
2026-06-28T09:04:00Z  ERROR  Microsoft.Web  SSL certificate binding failed for domain myapp.contoso.com: Certificate thumbprint not found in Key Vault.
""",
    "Azure DevOps Pipelines – Permission & Timeout Errors": """\
2026-06-28T11:00:00Z  ERROR  AzurePipelines  ##[error]The pipeline is not valid. Job Build/steps/AzureWebApp: Step AzureWebApp input ConnectedServiceName references service connection 'Azure-Prod' which could not be found. The service connection does not exist or has not been authorized for use.
2026-06-28T11:05:30Z  ERROR  AzurePipelines  ##[error]No hosted parallelism has been purchased or granted. To request a free parallelism grant, please fill out the following form https://aka.ms/azpipelines-parallelism-request
2026-06-28T11:10:00Z  ERROR  AzurePipelines  ##[error]Task 'AzureRmWebAppDeployment' failed: Timeout waiting for deployment to complete. Elapsed time: 00:30:00 (limit: 00:30:00).
2026-06-28T11:15:00Z  WARNING  AzurePipelines  ##[warning]Resource file lock timeout. Unable to acquire lock for artifact 'drop' after 5 minutes.
2026-06-28T11:20:00Z  ERROR  AzurePipelines  ##[error]The directory '/home/vsts/work/1/s' does not exist or is empty. Repository checkout may have failed.
""",
    "Azure SQL – Connection & Firewall Issues": """\
2026-06-28T07:00:00Z  ERROR  SqlException  Cannot open server 'mysqlserver' requested by the login. Client with IP address '20.10.5.100' is not allowed to access the server. To enable access, use the Azure Management Portal or run sp_set_firewall_rule on the master database.  ErrorCode: 40615
2026-06-28T07:01:00Z  ERROR  SqlException  Login failed for user 'sqladmin'. The password does not meet the complexity requirements. Error: 18456, Severity: 14, State: 8.
2026-06-28T07:05:00Z  ERROR  SqlException  Database 'mydb' on server 'mysqlserver' is not currently available. Please retry the connection later. ErrorCode: 40613
2026-06-28T07:10:00Z  WARNING  SqlAudit  DTU quota reached for server mysqlserver. Current DTU consumption: 100%. Throttling is in effect.
2026-06-28T07:15:00Z  ERROR  SqlException  The transaction log for database 'mydb' is full due to 'LOG_BACKUP'. ErrorCode: 9002
""",
}
