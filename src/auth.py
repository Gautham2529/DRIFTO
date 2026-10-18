"""Authentication helpers for the Streamlit dashboard.

Credentials are owned by the configured OpenID Connect provider (Auth0), not
stored or verified by Drifto. Auth0 can offer email/password registration and
the configured social connections from its hosted sign-in page.
"""
from __future__ import annotations

from collections.abc import Mapping

import streamlit as st


def auth_configuration_error(auth_settings: object) -> str | None:
    """Return a diagnostic for incomplete Auth0 settings, if any."""
    if not isinstance(auth_settings, Mapping):
        return "The [auth] configuration must be a TOML table."

    provider = auth_settings.get("auth0")
    if not isinstance(provider, Mapping):
        return "The [auth.auth0] provider configuration is missing."

    required_shared = ("redirect_uri", "cookie_secret")
    required_provider = ("client_id", "client_secret", "server_metadata_url")
    if any(
        not isinstance(auth_settings.get(key), str)
        or not auth_settings.get(key).strip()
        for key in required_shared
    ):
        return "The [auth] configuration is missing redirect_uri or cookie_secret."
    if any(
        not isinstance(provider.get(key), str) or not provider.get(key).strip()
        for key in required_provider
    ):
        return "The [auth.auth0] configuration is missing a required provider value."
    return None


def render_auth_gate() -> bool:
    """Require Auth0 login when configured; otherwise leave the demo public.

    Returns True when authentication is configured and a user is signed in.
    A present but incomplete auth configuration fails closed so a typo cannot
    accidentally expose an app that was intended to be private.
    """
    try:
        auth_settings = st.secrets["auth"]
    except (FileNotFoundError, KeyError):
        st.sidebar.caption("Public demo · sign-in is not configured")
        return False

    configuration_error = auth_configuration_error(auth_settings)
    if configuration_error:
        st.error(
            "Authentication is configured incompletely. "
            f"{configuration_error} The dashboard stays locked until this is corrected."
        )
        st.stop()

    if not st.user.is_logged_in:
        st.markdown(
            "<div class='hero'><span class='eyebrow'>DRIFTO ACCOUNT</span>"
            "<h1>Welcome to Drifto</h1>"
            "<p>Sign in or create an account to use the model drift dashboard.</p></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Sign in or create an account",
            type="primary",
            on_click=st.login,
            args=("auth0",),
        )
        st.caption(
            "The secure sign-in page provides email/password and any social "
            "providers enabled by the Drifto administrator."
        )
        st.stop()

    identity = st.user.get("email") or st.user.get("name") or "Signed-in user"
    st.sidebar.caption(f"Signed in as {identity}")
    st.sidebar.button("Sign out", on_click=st.logout)
    return True
