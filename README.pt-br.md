# Predição de Desempenho Institucional no ENAMED 2025

Este projeto investiga a capacidade preditiva de variáveis socioeconômicas e institucionais dos microdados do **ENAMED 2025** para estimar o desempenho acadêmico em nível de curso utilizando modelos de machine learning.

O ENAMED é uma modalidade especializada do exame nacional (ENADE) projetada especificamente para as faculdades de medicina no Brasil. Esta pesquisa explora o conjunto de dados para identificar padrões que influenciam a qualidade do ensino médico.

📄 Leia este README em Inglês: README.md

## Objetivo
Avaliar o potencial preditivo de variáveis agregadas institucionais e socioeconômicas dos microdados do ENAMED, com foco específico na **Universidade Federal de Juiz de Fora (UFJF)**.

## Perguntas de Pesquisa
- Quais fatores socioeconômicos impactam mais significativamente o desempenho médio dos cursos de medicina?
- Modelos de machine learning podem prever com precisão as categorias de desempenho dos cursos com base em perfis institucionais?
- Como o desempenho dos campi da UFJF se compara às tendências nacionais e regionais?

## Conjunto de Dados (Dataset)
Microdados do ENAMED 2025 fornecidos pelo INEP.
**Nota sobre a Metodologia:** Devido à conformidade com a LGPD (Lei Geral de Proteção de Dados), o conjunto de dados não permite o cruzamento de dados em nível de estudante individual. Portanto, este estudo emprega uma **abordagem em nível agregado**, agrupando os dados pelos Códigos de Curso (CO_CURSO).

Os dados brutos não estão incluídos neste repositório.

## Pipeline e Arquitetura do Projeto

O fluxo de trabalho está dividido em 6 notebooks sequenciais estruturados da seguinte forma:

* **01_preprocessing.ipynb:** Ingestão dos microdados brutos do ENAMED, agregação por `CO_CURSO` para conformidade com a LGPD e cálculo das métricas de desempenho.
* **02_feature_selection_and_regression.ipynb:** Análise preliminar explorando o impacto das variáveis agregadas nas notas gerais utilizando modelos de regressão.
* **03_exam_perception_classification.ipynb:** Investigação inicial utilizando as colunas de percepção da prova dos alunos como preditores para a categorização de desempenho.
* **04_feature_importance_and_multivariable_classification.ipynb:** Mapeamento global de importância de atributos (*feature importance*) utilizando modelos Random Forest base para filtrar indicadores socioeconômicos e institucionais.
* **05_feature_selection_and_model_optimization.ipynb:** Filtragem automática de ruído (redução de 60 para 20 variáveis-chave) e ajuste fino de hiperparâmetros com `GridSearchCV`. Alcançou uma **acurácia de 78% na validação cruzada** com as configurações otimizadas (`n_estimators: 50`, `max_depth: 6`, `criterion: 'entropy'`).
* **06_ufjf_insights_and_local_contributions.ipynb:** Desconstrução da "caixa-preta" do modelo utilizando a Análise de Contribuição Local de Atributos para extrair diagnósticos pedagógicos estratégicos e específicos para cada campus da UFJF (Juiz de Fora e Governador Valadares).

## Status do Projeto
**Concluído.** O pipeline de machine learning está totalmente estabelecido e os coeficientes institucionais foram traduzidos com sucesso em insights práticos de gestão.

## Principais Descobertas e Impacto Regional
* **UFJF Governador Valadares:** Classificado como *Alta Performance* com uma sólida **confiança de previsão de 94,6%**. Impulsionado fortemente por uma alta retenção de conteúdo (`I7_D`) e reconhecimento direto da contribuição do curso (`I9_A`).
* **UFJF Juiz de Fora:** Identificado corretamente como *Alta Performance*, mas estabiliza como um caso de fronteira (**57,6% de confiança**) devido a um sentimento interno misto dos alunos — equilibrando elogios à clareza da prova (`I4_B`) com atritos localizados gerados pela percepção de dificuldade de conteúdo (`I6_A` e `I1_D`).

## Painel de Auditoria Interativo (Dashboard)

Para traduzir os coeficientes matemáticos do pipeline em insights práticos para gestores educacionais, um painel web interativo foi desenvolvido utilizando o **Streamlit**. A interface funciona como uma plataforma de auditoria institucional moldada para a tomada de decisões pedagógicas.

### Principais Funcionalidades do Dashboard:
* **Estrutura de Busca Dinâmica:** Consulta instantânea de qualquer programa médico no Brasil filtrando pelo seu Código de Curso único (`CO_CURSO`).
* **Mapeamento de Contribuição de Atributos ao Vivo:** Gera gráficos de barras horizontais em tempo real mostrando os escores exatos de força de impacto local derivados do modelo Random Forest otimizado, reordenando os atributos dinamicamente a cada consulta.
* **Diagnósticos Pedagógicos Automatizados:** Analisa programaticamente as respostas dos questionários dos alunos para destacar pontos fortes localizados e gargalos curriculares críticos.
* **Seletor de Idioma Nativo:** Tradução completa da interface do usuário, gráficos e pacotes de dados entre **Inglês** e **Português** através de um controlador na barra lateral.
* **Conformidade LGPD-by-Design:** Construído inteiramente sobre identificadores agregados e mascarados para preservar a privacidade dos dados e evitar a reidentificação individual.

## Trabalhos Relacionados
Este projeto baseia-se em experiências anteriores com a análise de microdados do ENADE, migrando o foco para o recém-estabelecido exame médico (ENAMED) e sua matriz de avaliação específica.

Referências:
[https://github.com/Ivanylson/Ontology_ENADE](https://github.com/Ivanylson/Ontology_ENADE)

[https://github.com/lucasdahbar/enade-performance-prediction](https://github.com/lucasdahbar/enade-performance-prediction)
