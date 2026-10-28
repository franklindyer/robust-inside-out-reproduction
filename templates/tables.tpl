\begin{tabular}{llll}
& model & avg $K$ & avg $K^\ast$ \\
\hline
{% for int_ext in ["external", "internal"] %}
{{ int_ext -}}
{% for model_id in ["mistral-7b", "llama3-8b", "gemma2-9b"] %}
 & \verb|{{ model_id }}| & {{ '%.2f' | format(k_avgs[(model_id, "k_sampled_only", int_ext)]) }} & {{ '%.2f' | format(k_avgs[(model_id, "k_star_sampled_only", int_ext)]) }} \\
{%- endfor %}
\hline
{% endfor %}
\end{tabular}

\begin{tabular}{lll}
    Unicode block & Proportion answers with char & Decrease \\
    \hline
    {% for r in sampled_blocks %}
    \verb|{{ r["name"] }}| & {{ '%.4f' | format(r["once_prob"]) }} & \textcolor{red}{ -{{ '%.2f' |  format(100*r["per_dec"]) }}\% } \\
    {%- endfor %}
\end{tabular}
