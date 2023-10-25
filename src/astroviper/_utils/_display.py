def _dict_to_html(d, indent=0):
    from IPython.display import HTML

    html = ""
    for key, value in d.items():
        if isinstance(value, dict):
            html += f"<div style='margin-left: {indent}em;'><details><summary>{key}</summary>{_dict_to_html(value, indent + 1)}</details></div>"
        else:
            html += f"<div style='margin-left: {indent}em;'><strong>{key}:</strong> {value}</div>"
    return html
