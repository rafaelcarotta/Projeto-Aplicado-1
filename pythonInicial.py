import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

# =========================
# 1. DETECTAR INÍCIO DA TABELA
# =========================

def encontrar_inicio_tabela(df_raw, min_nao_nulos=8, janela=5):
    for i in range(len(df_raw) - janela):
        bloco = df_raw.iloc[i:i+janela]
        if (bloco.notna().sum(axis=1) >= min_nao_nulos).all():
            return i
    raise ValueError("Não encontrou início da tabela")

# =========================
# 2. COLUNAS ÚNICAS
# =========================

def garantir_colunas_unicas(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idx = cols[cols == dup].index
        cols[idx] = [f"{dup}_{i}" if i != 0 else dup for i in range(len(idx))]
    df.columns = cols
    return df

# =========================
# 3. CARREGAR DADOS
# =========================

def carregar_sinisa(caminho):
    df_raw = pd.read_excel(caminho, header=None)

    start = encontrar_inicio_tabela(df_raw)
    print(f"Início detectado: {start}")

    header = df_raw.iloc[start:start+2].fillna('')
    colunas = header.apply(lambda x: ' '.join(x.astype(str)).strip(), axis=0)

    colunas = (
        colunas
        .str.replace('\n', ' ')
        .str.replace('  ', ' ')
        .str.strip()
    )

    df = df_raw.iloc[start+2:].copy()
    df.columns = colunas
    df = garantir_colunas_unicas(df)
    df.reset_index(drop=True, inplace=True)

    return df

# =========================
# 4. LIMPEZA
# =========================

def limpar_df(df):
    df.columns = (
        df.columns
        .str.lower()
        .str.replace('\n', ' ')
        .str.replace(' ', '_')
        .str.replace('[^a-z0-9_]', '', regex=True)
    )

    df = df.replace(['', ' ', '-', 'nan'], np.nan)
    df = df.infer_objects(copy=False)

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace('.', '', regex=False)
                    .str.replace(',', '.', regex=False)
                )
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    return df

# =========================
# 5. IBGE
# =========================

def ajustar_ibge(df):
    for col in df.columns:
        serie = df[col].astype(str)
        if serie.str.match(r'^\d{7}$', na=False).sum() > 20:
            df.rename(columns={col: "codigo_ibge"}, inplace=True)
            df["codigo_ibge"] = serie.str.extract(r'(\d{7})')[0]
            print(f"IBGE: {col}")
            return df

    raise ValueError("IBGE não encontrado")

# =========================
# 6. PIPELINE
# =========================

def processar(caminho):
    df = carregar_sinisa(caminho)
    df = limpar_df(df)
    df = ajustar_ibge(df)
    return df

# =========================
# 7. CARREGAR ARQUIVOS
# =========================

df_fin = processar('SINISA_AGUASPLUVIAIS_Informacoes_Formularios_Financeiros.xlsx')
df_inf = processar('SINISA_AGUASPLUVIAIS_Informacoes_Formularios_Infraestruturas.xlsx')
df_tec = processar('SINISA_AGUASPLUVIAIS_Informacoes_Formularios_Tecnicos.xlsx')

# garantir tipo
for df in [df_fin, df_inf, df_tec]:
    df["codigo_ibge"] = df["codigo_ibge"].astype(str)

# =========================
# 8. MERGE
# =========================

df_final = df_fin.merge(df_inf, on="codigo_ibge", how="outer")
df_final = df_final.merge(df_tec, on="codigo_ibge", how="outer")

# =========================
# 9. LIMPEZA FINAL
# =========================

df_final = df_final.dropna(axis=1, how='all')
df_final = df_final.loc[:, df_final.isnull().mean() < 0.9]

# =========================
# 10. DETECTAR COLUNAS AUTOMATICAMENTE
# =========================

def detectar_coluna_numerica(df, minimo_valores_validos=100):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().sum() > minimo_valores_validos:
                return col
    return None

# população (prioridade por nome)
col_pop = None
for col in df_final.columns:
    if "pop" in col:
        col_pop = col
        break

# despesa (primeira numérica relevante)
col_despesa = detectar_coluna_numerica(df_final)

print("População:", col_pop)
print("Despesa:", col_despesa)

# =========================
# 11. KPI
# =========================

if col_pop and col_despesa:
    df_final = df_final[df_final[col_pop] > 0]

    df_final["investimento_per_capita"] = (
        df_final[col_despesa] / df_final[col_pop]
    )

    print("KPI criado!")

else:
    print("Não foi possível criar KPI")

# =========================
# 12. EXPORTAR
# =========================

df_final.to_csv("base_tratada_sinisa.csv", index=False)

# =========================
# 13. OUTPUT
# =========================

print("\nFINAL:", df_final.shape)
print(df_final.head())

if "investimento_per_capita" in df_final.columns:
    print("\nTOP 10:")
    print(df_final[["codigo_ibge", "investimento_per_capita"]]
          .sort_values(by="investimento_per_capita", ascending=False)
          .head(10))