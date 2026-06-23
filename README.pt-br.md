# ENAMED 2025 — Predição de Desempenho Institucional & Hub de Auditoria

Este repositório investiga a capacidade preditiva de variáveis socioeconômicas e institucionais dos microdados do **ENAMED 2025** para estimar o desempenho acadêmico a nível de curso utilizando modelos de aprendizado de máquina.

Este projeto acompanha dois artigos complementares (*companion papers*) atualmente sob revisão: um estudo de Mineração de Dados Educacionais (*Educational Data Mining*) e um estudo de arquitetura de software voltado para o suporte à decisão institucional.

📄 Leia este README em Inglês: README.md

## Objetivo
Avaliar o potencial preditivo de variáveis agregadas institucionais e socioeconômicas dos microdados do ENAMED, com foco específico na **Universidade Federal de Juiz de Fora (UFJF)**.

## Perguntas de Pesquisa
- Quais fatores socioeconômicos impactam mais significativamente o desempenho médio dos cursos de medicina?
- Modelos de aprendizado de máquina conseguem prever com precisão as categorias de desempenho dos cursos com base nos perfis institucionais?
- Como o desempenho dos campi da UFJF se compara às tendências nacionais e regionais?

## Base de Dados & Fonte dos Dados
Microdados do ENAMED 2025 fornecidos pelo INEP.
**Nota sobre a Metodologia:** Devido à conformidade com a LGPD (Lei Geral de Proteção de Dados), a base de dados não permite o cruzamento de dados a nível individual do estudante. Portanto, este estudo emprega uma **abordagem a nível agregado**, agrupando os dados pelos Códigos de Curso (`CO_CURSO`). Os dados brutos não estão incluídos neste repositório.

* **Repositório Oficial do INEP:** Link para o site do governo com os microdados do ENAMED 2025: [Repositório de Dados Abertos do INEP](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enamed)

## Pipeline do Projeto & Arquitetura Core

O fluxo de trabalho está dividido em 6 notebooks principais sequenciais e uma camada avançada de expansão estatística/generativa:

### Pipeline Principal (Notebooks)
* **01_preprocessing.ipynb:** Ingestão dos microdados brutos do ENAMED, agregação por `CO_CURSO` em conformidade com a LGPD e cálculo das métricas de desempenho.
* **02_feature_selection_and_regression.ipynb:** Análise preliminar explorando o impacto das variáveis agregadas nas notas gerais utilizando modelos de regressão.
* **03_exam_perception_classification.ipynb:** Investigação inicial sobre as colunas de percepção dos estudantes a respeito da prova como preditoras para a categorização de desempenho.
* **04_feature_importance_and_multivariable_classification.ipynb:** Mapeamento global de importância das variáveis utilizando modelos baseline de Random Forest para filtrar os indicadores socioeconômicos e institucionais.
* **05_feature_selection_and_model_optimization.ipynb:** Filtragem automática de ruído (redução de 60 para 20 variáveis-chave) e ajuste de hiperparâmetros usando `GridSearchCV`. Alcançou uma **acurácia de 78% na validação cruzada** com as configurações otimizadas (`n_estimators: 50`, `max_depth: 6`, `criterion: 'entropy'`).
* **06_ufjf_insights_and_local_contributions.ipynb:** Desconstrução do modelo caixa-preta utilizando Análise de Contribuição de Atributos Locais para extrair diagnósticos pedagógicos estratégicos e específicos para os campi da UFJF (Juiz de Fora e Governador Valadares).

### Validação Estatística Avançada (`analysis/`)
* **analysis_extras.py:** Motor estatístico robusto que executa a comparação de 5 classificadores sob validação cruzada estratificada em 5 folds; teste de significância binomial unicaudal contra o nível de acaso (+ IC de 95% de Wilson); mapeamento da *curva do paradoxo da complexidade* (acurácia vs. número de variáveis); e análise de erros nas fronteiras dos dados.

## Principais Descobertas & Impacto Regional
* **UFJF Governador Valadares:** Classificado como *Alto Desempenho* com uma sólida **confiança de predição de 94.6%**. Fortemente impulsionado por uma boa retenção de conteúdo (`I7_D`) e valorização direta do curso (`I9_A`).
* **UFJF Juiz de Fora:** Identificado corretamente como *Alto Desempenho*, mas se estabiliza como um caso de fronteira (**57.6% de confiança**) devido ao sentimento interno misto dos estudantes — equilibrando elogios à clareza do exame (`I4_B`) com fricções localizadas causadas pela percepção de dificuldade do conteúdo (`I6_A` e `I1_D`).

## Dashboard de Auditoria Interativo & Camada LLM

Para traduzir os coeficientes matemáticos do pipeline em insights acionáveis para gestores educacionais, um Web Dashboard interativo foi desenvolvido utilizando o **Streamlit**.

### Principais Funcionalidades do Dashboard:
* **Estrutura de Busca Dinâmica:** Consulta instantânea de qualquer programa de medicina do Brasil filtrando pelo seu Código de Curso único (`CO_CURSO`).
* **Mapeamento de Contribuição de Variáveis em Tempo Real:** Gera gráficos de barras horizontais dinamicamente, mostrando a força exata do impacto local derivado do modelo Random Forest otimizado, reordenando as variáveis a cada nova consulta.
* **Diagnósticos Pedagógicos Automatizados:** Analisa programaticamente as respostas dos questionários dos estudantes para destacar pontos fortes e gargalos críticos na estrutura curricular.
* **Camada de Explicação por LLM Aterrada (`llm_explainer.py`):** Uma camada opcional e segura de explicação em linguagem natural alimentada pelo Gemini (`gemini-3.5-flash`). Processa APENAS os números pré-calculados para gerar resumos textuais para os coordenadores, garantindo total privacidade dos dados com um fallback baseado em regras caso o serviço esteja offline.
* **Alternador de Idioma Nativo:** Tradução completa da interface do usuário, gráficos e pacotes de dados entre **Inglês** e **Português** através de um controle na barra lateral.

---

## Primeiros Passos & Como Executar

### 1. Configuração do Ambiente
Clone este repositório e certifique-se de ter as bibliotecas necessárias instaladas:

```bash
git clone [https://github.com/lucasdahbar/enamed-performance-prediction.git](https://github.com/lucasdahbar/enamed-performance-prediction.git)
cd enamed-performance-prediction
pip install pandas numpy scikit-learn scipy matplotlib google-genai streamlit
```

### 2. Executando as Análises Estatísticas
Execute o script de validação apontando para o caminho dos seus dados brutos para imprimir os benchmarks e exportar as métricas (results/results.json e results/accuracy_vs_k.png):

```Bash
python analysis/analysis_extras.py --data data/raw/microdados_enade_2025_arq3.txt --outdir results --scan-seeds
```

### 3. Inicializando o Dashboard (Com Suporte a LLM)
Para habilitar a aba de explicações generativas, gere uma chave de API no Google AI Studio e coloque-a dentro de .streamlit/secrets.toml:

```Ini, TOML
GEMINI_API_KEY = "sua-chave-api-real-aqui"
```

Em seguida, inicie o servidor do Streamlit utilizando a flag explícita de módulo do Python:

```Bash
python -m streamlit run app.py
```

### Trabalhos Relacionados
Este projeto baseia-se em experiências anteriores com a análise de microdados do ENADE, migrando o foco para o recém-estabelecido exame médico (ENAMED) e sua matriz de avaliação específica.

Referências:

https://github.com/Ivanylson/Ontology_ENADE

https://github.com/lucasdahbar/enade-performance-prediction