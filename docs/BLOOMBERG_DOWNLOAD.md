# Guia: Download de dados Bloomberg no lab do Insper

**Objetivo:** extrair os dados que destravam o teto do sistema (survivorship
bias, PEAD real, consenso de analistas) e commitá-los em `bloomberg_data/`
para o pipeline consumir.

**Tempo estimado:** 45–90 min na primeira vez (inclui descobrir o ambiente).

---

## Antes de ir (checklist)

- [ ] Conta GitHub logável no navegador do lab (ou pendrive como plano B)
- [ ] Este repositório acessível (clone público: `git clone https://github.com/danielgozzifernandes-ctrl/teste-financeiro.git`)
- [ ] Saber o login do terminal Bloomberg do lab (geralmente já logado ou via conta institucional Insper)

## Passo 1 — Descobrir o ambiente (5 min, só na primeira vez)

No PC do lab com Bloomberg Terminal aberto, descubra qual caminho usar:

1. Abra o prompt de comando (`Win+R` → `cmd`) e teste:
   ```
   python --version
   pip --version
   ```
2. **Se tem Python:** teste `pip install xbbg` (precisa do Terminal logado
   na mesma máquina). Se instalar, use o **Caminho A** (script pronto).
3. **Se NÃO tem Python ou pip é bloqueado:** use o **Caminho B** (Excel),
   que funciona em qualquer máquina com Terminal + Office.

## Caminho A — Script Python (preferido)

1. Clone o repo (ou copie só `tools/bloomberg_download.py` via pendrive):
   ```
   git clone https://github.com/danielgozzifernandes-ctrl/teste-financeiro.git
   cd teste-financeiro
   pip install xbbg pandas
   ```
2. Rode o script (Terminal Bloomberg precisa estar LOGADO na máquina):
   ```
   python tools/bloomberg_download.py
   ```
   Ele cria `bloomberg_data/*.csv` com data no nome. Cada bloco é
   independente: se um campo falhar (permissão/limite), os outros seguem.
3. Commite e suba:
   ```
   git add bloomberg_data/
   git commit -m "data: bloomberg download YYYY-MM-DD"
   git push
   ```
   Sem git no lab? Copie a pasta `bloomberg_data/` para o pendrive/OneDrive
   e commite de casa.

## Caminho B — Excel (fallback universal)

No Excel do lab (add-in Bloomberg ativo), monte uma planilha por dataset e
exporte como CSV (`Salvar como → CSV UTF-8`):

### B.1 — Composição histórica do IBX (PRIORIDADE #1)

No Terminal: `IBX Index MEMB <GO>` mostra os membros atuais. Para histórico,
use no Excel:
```
=BDS("IBX Index", "INDX_MWEIGHT_HIST", "END_DATE_OVERRIDE", "20160101")
```
Repita para datas trimestrais (jan/abr/jul/out de cada ano, 2016→hoje).
Salve como `ibx_composition_YYYYMMDD.csv` — uma coluna `date`, uma `ticker`,
uma `weight`.

### B.2 — Consenso de EPS (PEAD real)

Para cada ticker do universo (lista em `data/universe.csv`):
```
=BDP("PETR4 BZ Equity", "BEST_EPS")           ← consenso próximo tri
=BDP("PETR4 BZ Equity", "IS_EPS")             ← último reportado
=BDP("PETR4 BZ Equity", "BEST_EPS_NUMEST")    ← nº de estimativas
```
Salve como `consensus_eps.csv` com colunas: `ticker, best_eps, reported_eps, n_estimates`.

### B.3 — Preço-alvo com dispersão

```
=BDP("PETR4 BZ Equity", "BEST_TARGET_PRICE")
=BDP("PETR4 BZ Equity", "BEST_TARGET_HI")
=BDP("PETR4 BZ Equity", "BEST_TARGET_LO")
=BDP("PETR4 BZ Equity", "TOT_ANALYST_REC")
```
Salve como `analyst_targets.csv`.

### B.4 — Short interest (bônus)

```
=BDP("PETR4 BZ Equity", "SI_TOT_EQY")
=BDP("PETR4 BZ Equity", "SHORT_INT_RATIO")
```
Salve como `short_interest.csv`.

> Dica: monte a coluna A com os ~97 tickers (formato `XXXX4 BZ Equity`) e
> arraste as fórmulas — o Excel resolve tudo de uma vez.

## Passo 3 — Formato esperado em `bloomberg_data/`

```
bloomberg_data/
├── ibx_composition.csv      # date,ticker,weight  (todas as datas empilhadas)
├── consensus_eps.csv        # ticker,best_eps,reported_eps,n_estimates,asof_date
├── analyst_targets.csv      # ticker,target_mean,target_hi,target_lo,n_analysts,asof_date
└── short_interest.csv       # ticker,si_total,si_ratio,asof_date
```

Sempre inclua uma coluna `asof_date` (data do download) — sem ela o dado
não é point-in-time e perde metade do valor.

## Passo 4 — De volta em casa

Peça ao Claude: *"os CSVs da Bloomberg estão em bloomberg_data/, integre"*.
O plano de integração já existe (módulo `src/bloomberg_data.py` plugável com
fallback para yfinance/brapi; ver memória do projeto). A integração é ~1 sessão.

## Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `xbbg` conecta mas retorna vazio | Terminal não logado / sessão expirada | Logar no Terminal e re-rodar |
| `#N/A Authorization` no Excel | Campo fora da licença do lab | Pular o campo; anotar qual |
| `INDX_MWEIGHT_HIST` recusa datas antigas | Limite da licença educacional | Pegar o máximo que der (mesmo 5 anos já ajuda) |
| Limite diário de dados atingido | Licenças edu têm cota | Priorizar B.1 > B.2 > B.3 > B.4; voltar outro dia |
