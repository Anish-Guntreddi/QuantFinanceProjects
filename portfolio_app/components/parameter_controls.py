"""Dynamic parameter controls built from strategy_card.json."""

import streamlit as st


def render_controls(interactive_params):
    """Build Streamlit widgets from interactive_params list. Returns dict of param values."""
    params = {}
    if not interactive_params:
        st.info("No interactive parameters for this strategy.")
        return params

    for p in interactive_params:
        name = p.get("name", "param")
        label = p.get("label", name)
        ptype = p.get("type", "slider")

        if ptype == "slider":
            params[name] = st.slider(
                label,
                min_value=float(p.get("min", 0)),
                max_value=float(p.get("max", 1)),
                value=float(p.get("default", 0.5)),
                step=float(p.get("step", 0.01)),
                key=f"param_{name}",
            )
        elif ptype == "selectbox":
            options = p.get("options", [])
            default_idx = options.index(p.get("default")) if p.get("default") in options else 0
            params[name] = st.selectbox(label, options, index=default_idx, key=f"param_{name}")
        elif ptype == "number_input":
            params[name] = st.number_input(
                label,
                min_value=float(p.get("min", 0)),
                max_value=float(p.get("max", 100)),
                value=float(p.get("default", 1)),
                key=f"param_{name}",
            )

    return params
