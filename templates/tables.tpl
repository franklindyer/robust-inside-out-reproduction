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
 model & external $K^\ast$ & internal $K^\ast$ \\
\hline
{% for model_id in ["mistral-7b", "llama3-8b", "gemma2-9b"] %}
 \verb|{{ model_id }}| & {{ '%.2f' | format(k_star_unsampled[(model_id, "external")]) }} & {{ '%.2f' | format(k_star_unsampled[(model_id, "internal")]) }} \\
{%- endfor %}
\hline
\end{tabular}

{% macro ranking_table(cs, num) %}
\begin{tabular}{ll|ll}
\multicolumn{4}{c}{ "{{ cs['question'] }}" } \\
external top answers & scores & internal top answers & scores \\
\hline
{% for i in range(num) -%}
{% set r_ext = cs['top10_ext'][i] %}{% set r_int = cs['top10_int'][i] -%}
{% set color_ext = "\\cellcolor{green!30}" if r_ext['correct'] else "\\cellcolor{red!30}" %}
{% set color_int = "\\cellcolor{green!30}" if r_int['correct'] else "\\cellcolor{red!30}" %}
{{ r_ext["text"] | replace("&", "\\&") }} {{ color_ext }} 
    & {{ '%.3f' | format(r_ext["score"]) }} {{ color_ext }}
    & {{ r_int["text"] | replace("&", "\\&") }} {{ color_int }}
    & {{ '%.3f' | format(r_int["score"]) }} {{ color_int }} \\
{% endfor %}
\hline
 $K = $ {{ '%.2f' | format(cs["k_ext"]) }} & & $K = $ {{ '%.2f' | format(cs["k_int"]) }} & 
\end{tabular}
{% endmacro %}

{{ ranking_table(case_studies["llama3-8b"]["1056"], 7) }}
{{ ranking_table(case_studies["gemma2-9b"]["6519"], 7) }}

\begin{tabular}{llll}
  question ID & subject & relation & object \\
\hline
{% for qid, r in rewardbert_data["reflexivity_violation"].items() -%}
 {{ qid }} & {{ r["subject"] | replace("&", "\\&") }} & \verb|{{ r["relation"] }}| & {{ r["object"] }} \\
{% endfor %}
\hline
\end{tabular}

\begin{tabular}{lll}
    Unicode block & Proportion answers with char & Decrease \\
    \hline
    {% for r in sampled_blocks %}
    \verb|{{ r["name"] }}| & {{ '%.4f' | format(r["once_prob"]) }} & \textcolor{red}{ -{{ '%.2f' |  format(100*r["per_dec"]) }}\% } \\
    {%- endfor %}
\end{tabular}
