# Download de dados Bloomberg no lab do Insper

O que só a Bloomberg dá de forma prática, e o que o sistema mais precisa:

1. composição histórica de IBOV/IBX/IBrA → universo sem viés de sobrevivência;
2. histórico de divulgações de resultado com EPS reportado e esperado → surpresa (PEAD) de verdade;
3. histórico de consenso (EPS, dispersão, preço-alvo, recomendações) → revisões;
4. preços e retorno total dos ex-membros, inclusive deslistados.

Fundamentos contábeis não entram aqui: a CVM (DFP/ITR com data de entrega)
dá isso point-in-time e de graça.

Tempo: 1–2 h na primeira vez. Licença acadêmica tem cota de dados, então rode
o piloto primeiro e as etapas em ordem de prioridade.

## Antes de ir

- [ ] Login do Terminal do lab (geralmente conta institucional)
- [ ] Repositório acessível pelo navegador (Code → Download ZIP) ou um pendrive com `tools/bloomberg_download.py` e `data/universe.csv` na mesma estrutura de pastas
- [ ] Pendrive ou OneDrive para trazer os CSVs de volta

## Caminho A — script (preferido)

O script precisa rodar na mesma máquina em que o Terminal está logado.

```
pip install xbbg pandas
python tools/bloomberg_download.py --pilot
```

O piloto baixa 5 tickers, só o IBOV e 2 datas em `bloomberg_data/pilot/`.
Se algum campo falhar, conferir o nome com `FLDS <GO>` e ajustar a constante
no topo do script. Depois:

```
python tools/bloomberg_download.py --steps comp,earn
python tools/bloomberg_download.py --steps cons
python tools/bloomberg_download.py --steps px
python tools/bloomberg_download.py --steps si      # se sobrar cota
```

`comp` precisa rodar antes das outras: é dela que saem os ex-membros que as
etapas seguintes incluem.

## Caminho B — Excel (sem Python no lab)

Uma planilha por dataset, exportada como CSV UTF-8.

| Dataset | Fórmula |
|---|---|
| Composição | `=BDS("IBOV Index","INDX_MWEIGHT_HIST","END_DATE_OVERRIDE","20260915")` — repetir para o dia 15 de cada mês desde 2010, e para `IBX Index` e `IBRA Index` |
| Resultados | `=BDS("PETR4 BZ Equity","EARN_ANN_DT_TIME_HIST_WITH_EPS")` |
| Consenso | `=BDH("PETR4 BZ Equity","BEST_EPS,BEST_EPS_NUMEST,BEST_EPS_STDDEV,BEST_TARGET_PRICE,BEST_ANALYST_RATING","20100101","","Per=W","BEST_FPERIOD_OVERRIDE=1BF")` |
| Preços | `=BDH("PETR4 BZ Equity","PX_LAST,TOT_RETURN_INDEX_GROSS_DVDS,EQY_SH_OUT,TURNOVER","20100101","")` |

As carteiras do IBOV/IBX mudam na 1ª segunda-feira de janeiro, maio e
setembro. Amostrar todo dia 15 pega cada carteira nova.

## Formato de saída (`bloomberg_data/`)

```
index_composition.csv        index, date, membro, peso
earnings_history.csv        ticker, data de divulgação, EPS reportado/esperado
consensus_weekly.csv        date, ticker, campos de consenso
prices_daily.csv            date, ticker, preço, retorno total, ações, giro
short_interest_weekly.csv   date, ticker, short interest
```

Todos levam `asof_date`.

## Licença

Dado bruto da Bloomberg não pode ser redistribuído. `bloomberg_data/` está no
`.gitignore`: os CSVs ficam na máquina local e só derivados agregados entram
no repositório.

## Depois

Falta o loader (`src/bloomberg_data.py`) que lê esses arquivos e cai para
COTAHIST/CVM/yfinance quando um campo não existe.

## Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `xbbg` conecta mas volta vazio | Terminal não logado ou sessão expirada | Logar e rodar de novo |
| `#N/A Authorization` | Campo fora da licença do lab | Pular o campo e anotar |
| `INDX_MWEIGHT_HIST` recusa datas antigas | Limite da licença | Pegar o que der |
| Cota diária atingida | Licença acadêmica | Seguir a ordem de prioridade; voltar outro dia |
| Ticker de deslistada não resolve | Código antigo | Usar o ISIN/ID que vem na composição |
