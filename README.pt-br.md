# Previsão de Desempenho Institucional - ENAMED 2025

Este projeto investiga a capacidade preditiva de variáveis socioeconômicas e institucionais dos microdados do **ENAMED 2025** para estimar o desempenho acadêmico a nível de curso, utilizando modelos de machine learning.

O ENAMED é uma modalidade especializada do exame nacional (ENADE), projetada especificamente para as faculdades de medicina no Brasil. Esta pesquisa explora o conjunto de dados para identificar padrões que influenciam a qualidade da educação médica.

📄 Leia este README em Inglês: README.md

## Objetivo
Avaliar o potencial preditivo de variáveis agregadas institucionais e socioeconômicas dos microdados do ENAMED, com foco específico na **Universidade Federal de Juiz de Fora (UFJF)**.

## Perguntas de Pesquisa
- Quais fatores socioeconômicos impactam mais significativamente o desempenho médio dos cursos de medicina?
- Modelos de machine learning conseguem prever com precisão categorias de desempenho dos cursos com base em perfis institucionais?
- Como o desempenho dos campi da UFJF se compara às tendências nacionais e regionais?

## Dataset
Microdados do ENAMED 2025 fornecidos pelo INEP.  
**Nota sobre a Metodologia:** Devido à conformidade com a LGPD (Lei Geral de Proteção de Dados), o conjunto de dados não permite o cruzamento individual por aluno. Portanto, este estudo utiliza uma **abordagem em nível agregado**, agrupando os dados pelos Códigos de Curso (`CO_CURSO`).

Os dados brutos não estão incluídos neste repositório.

## Metodologia
- Filtragem de dados (Focando nos códigos da UFJF: `13103` e `5001167`)
- Pré-processamento de dados e agregação de características por curso
- Aprendizado supervisionado (classificação e regressão)
- Modelos de baseline: Random Forest
- Métricas de avaliação: Acurácia, R-squared e F1-score

## Status do Projeto
Configuração inicial e filtragem de dados.

## Trabalhos Relacionados
Este projeto baseia-se em experiências anteriores com a análise de microdados do ENADE, migrando o foco para o recém-estabelecido exame médico (ENAMED) e sua matriz de avaliação específica (Clínica Médica, Cirurgia, Pediatria, etc.).

Referências:
[https://github.com/Ivanylson/Ontology_ENADE](https://github.com/Ivanylson/Ontology_ENADE)
[https://github.com/lucasdahbar/enade-performance-prediction](https://github.com/lucasdahbar/enade-performance-prediction)
