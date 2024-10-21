import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import streamlit as st


# Seite "Signifikanztest"
def signifikanztest_page(dfGrouped):
    st.title('Signifikanztest')
    st.markdown("""
    ### Zeitdifferenz vs. Wolkenabdeckungsdaten
    """)
    # Wähle die abhängigen Variablen (Messwerte), die du analysieren möchtest
    dependent_vars = ['cloud_cover_diff', 'cloud_cover_low_diff', 'cloud_cover_mid_diff', 'cloud_cover_high_diff']

    # Erstelle eine Korrelationstabelle und zeige sie an
    st.subheader('Korrelationsmatrix')
    correlation_matrix = dfGrouped.corr()
    st.write(correlation_matrix)

    # Lineare Regression für jede aggregierte Spalte
    X = dfGrouped['timediffofforecast']  # Unabhängige Variable
    X = sm.add_constant(X)  # Konstanten-Term hinzufügen für den Intercept

    for var in selected_measurements:  #dependent_vars:
        y = dfGrouped[var]  # Abhängige Variable

        # OLS-Regression (ordinary least squares) durchführen
        model = sm.OLS(y, X).fit()

        # Regressionsergebnisse anzeigen
        st.subheader(f"Regressionsergebnisse für {var}")
        st.text(model.summary())  # Textform der Regressionsergebnisse

        # Regressionslinie plotten und anzeigen
        fig, ax = plt.subplots()
        sns.regplot(x='timediffofforecast', y=var, data=dfGrouped, line_kws={"color": "red"}, ax=ax)
        ax.set_title(f'Regression von {var} gegen timediffofforecast')
        st.pyplot(fig)

        # Pearson test
        # Pearson-Korrelation berechnen
        corr, p_value = pearsonr(dfGrouped['timediffofforecast'], dfGrouped[var])
        st.subheader(f"PEARSON-Korrelation für {var}:")
        st.text(f"Korrelation: {corr} mit p={p_value}")  # Textform der Regressionsergebnisse

        #print(f"Pearson-Korrelationskoeffizient für {var}: {corr}")
        #print(f"p-Wert: {p_value}\n")



# Upload-Funktion für die CSV-Datei
st.title("Wolkendeckenanalyse")
uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei aus", type=["csv"])

if uploaded_file is not None:
    # Daten laden
    df = pd.read_csv(uploaded_file)

    # Auswahl der Orte
    locations = df['locname'].unique()
    selected_locations = st.multiselect("Wählen Sie Orte aus:", locations.tolist(), default=locations[:2])

    # Filter auf ausgewählte Locations anwenden
    filtered_df = df[df['locname'].isin(selected_locations)]

    # Gruppierung nach 'locname' und 'timediffofforecast', Berechnung der Mittelwerte für die gewünschten Spalten
    dfDataGrouped = filtered_df.groupby(['locname', 'timediffofforecast']).agg({
        'cloud_cover_diff': 'mean',
        'cloud_cover_low_diff': 'mean',
        'cloud_cover_mid_diff': 'mean',
        'cloud_cover_high_diff': 'mean'
    }).reset_index()

    # Streamlit-Seiten verwenden
    page = st.sidebar.selectbox("Wählen Sie eine Seite", ["Plot", "Statistiken", "Signifikanztest I - Korrelation, Pearson", "Signifikanztest II - t-Test", "Tabellen"])

    # Gemeinsame Messreihen-Auswahl
    measurements = ['cloud_cover_diff', 'cloud_cover_low_diff', 'cloud_cover_mid_diff', 'cloud_cover_high_diff']
    selected_measurements = st.sidebar.multiselect("Wählen Sie Messreihen aus:", measurements, default=measurements)

    # Seite 1: Plot anzeigen
    if page == "Plot":
        # Optionen für Tendenzlinien und Prognosebereiche
        show_trend_lines = st.checkbox("Tendenzlinien anzeigen", value=True)
        show_forecast_area = st.checkbox("Prognosebereich anzeigen", value=True)

        # Diagramm erstellen
        fig, ax = plt.subplots(figsize=(10, 6))

        # Farbschema für die verschiedenen Messreihen
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_locations)))

        print(f"**** locations:{selected_locations}")

        # Linien für jede Messreihe und jeden Ort zeichnen
        for i, location in enumerate(selected_locations):
            loc_data = dfDataGrouped[dfDataGrouped['locname'] == location]
            
            # Messreihen zeichnen
            for measurement in selected_measurements:
                if measurement in ['cloud_cover_diff', 'cloud_cover_low_diff', 'cloud_cover_mid_diff', 'cloud_cover_high_diff']:
                    linestyle = '-' if measurement == 'cloud_cover_diff' else ('--' if measurement == 'cloud_cover_low_diff' else ('-.' if measurement == 'cloud_cover_mid_diff' else ':'))
                    ax.plot(loc_data['timediffofforecast'], loc_data[measurement], label=f'{location} - {measurement}', color=colors[i], linestyle=linestyle)

                    # Tendenzlinien zeichnen, nur wenn aktiviert
                    if show_trend_lines:
                        x = loc_data['timediffofforecast'].values.reshape(-1, 1)  # Feature Matrix
                        y = loc_data[measurement].values
                        model = LinearRegression()  # Model-Instanz nur hier erstellen
                        model.fit(x, y)
                        trend_line = model.predict(x)
                        ax.plot(loc_data['timediffofforecast'], trend_line, color=colors[i], linestyle=linestyle)

                    # Prognosebereich zeichnen, unabhängig von der Sichtbarkeit der Tendenzlinien
                    if show_forecast_area:
                        x = loc_data['timediffofforecast'].values.reshape(-1, 1)
                        y = loc_data[measurement].values
                        model = LinearRegression()  # Model-Instanz für Prognosebereich erstellen
                        model.fit(x, y)
                        # Prognosepunkte für den Bereich
                        forecast_x = np.arange(loc_data['timediffofforecast'].max() + 1, loc_data['timediffofforecast'].max() + 1 + int(0.3 * len(loc_data)), 1).reshape(-1, 1)
                        forecast_y = model.predict(forecast_x)
                        ax.fill_between(forecast_x.flatten(), forecast_y - 5, forecast_y + 5, color=colors[i], alpha=0.2)  # Prognosebereich

        # Achsenbeschriftungen und Titel
        ax.set_xlabel('Stunden zum Forecast')
        ax.set_ylabel('Wolkenbedeckung Unterschied (%)')
        ax.set_title('Liniendiagramm der Wolkendeckenunterschiede für mehrere Orte')

        # Legende hinzufügen
        ax.legend()

        # Diagramm anzeigen
        st.pyplot(fig)

    # Seite 2: Statistiken anzeigen
    elif page == "Statistiken":
        if len(selected_measurements) == 0:
            st.warning("Es wurden keine Messreihen ausgewählt. Bitte wählen Sie mindestens eine Messreihe aus.")
        else:

            # Grundstatistiken für nicht-aggregierte Daten berechnen und anzeigen
            statistics1 = filtered_df[selected_measurements].describe().T
            statistics1['IQR'] = statistics1['75%'] - statistics1['25%']
            statistics1['median'] = filtered_df[selected_measurements].median()  # Median manuell hinzufügen
            statistics1 = statistics1[['min', '25%', 'median', 'mean', '75%', 'max', 'std', 'IQR']]

            st.markdown("""
            # Grundstatistiken I
            **Nicht aggregierte Daten**
            """)
            st.dataframe(statistics1)

            # Grundstatistiken für aggregierte Daten berechnen und anzeigen
            statistics2 = dfDataGrouped[selected_measurements].describe().T
            statistics2['IQR'] = statistics2['75%'] - statistics2['25%']
            statistics2['median'] = dfDataGrouped[selected_measurements].median()  # Median manuell hinzufügen
            statistics2 = statistics2[['min', '25%', 'median', 'mean', '75%', 'max', 'std', 'IQR']]

            st.markdown("""
            # Grundstatistiken II
            **Aggregierte Daten (Mittelwerte der Messreihen)**
            """)
            st.dataframe(statistics2)


            ##########################################################################################
            ###### 24h - Klassen Statistik

            # Erstellen von 24-Stunden-Klassen basierend auf 'timediffofforecast'
            bins = np.arange(-24, df['timediffofforecast'].max() + 24, 24)  # Erstellen der Intervalle
            labels = [f'{i} bis {i + 23}' for i in bins[:-1]]  # Labels für die Intervalle erstellen

            #bins = [-float('inf'), 0, 24, 48, 72, 96, 120, 144, float('inf')]
            #labels = [0, 1, 2, 3, 4, 5, 6, 7]  # Anpassung der Labels nach Bedarf

            # Neue Spalte mit pd.cut() erstellen
            df['Diff_Prog_vs_Real_24h_Class'] = pd.cut(df['timediffofforecast'], bins=bins, labels=labels, right=False)
            #df.to_csv("____exp.csv")

            # Daten nach 'Diff_Prog_vs_Real_24h_Class' und 'locname' gruppieren und beschreiben
            grouped = df.groupby(['locname', 'Diff_Prog_vs_Real_24h_Class'])[selected_measurements].describe()

            # Sicherstellen, dass '75%' und '25%' existieren
            if '75%' in grouped.columns and '25%' in grouped.columns:
                grouped['IQR'] = grouped['75%'] - grouped['25%']  # IQR berechnen

            # Median separat für jede Messreihe berechnen
            #medians = df.groupby(['locname', 'Diff_Prog_vs_Real_24h_Class'])[selected_measurements].median()

            # Medians an den beschriebenen DataFrame anhängen
            #for measurement in selected_measurements:
            #    grouped[(measurement, 'median')] = medians[measurement]

            # Anzeige der Statistiken
            st.markdown("""
            # Basisstatistiken pro 24-Stunden-Intervall
            **Pro Messreihe und Ort**
            """)
            # Filtere den grouped DataFrame nach den ausgewählten Orten
            if selected_locations:
                filtered_grouped = grouped[grouped.index.get_level_values('locname').isin(selected_locations)]
            else:
                filtered_grouped = grouped

            filtered_grouped.to_csv("_filtered_grouped.csv")
            # Anzeige des gefilterten DataFrames
            st.dataframe(filtered_grouped)

    # Seite 3: Tabellenansicht anzeigen
    #if page == 'Signifikanztest I':
    if 'Signifikanztest I -' in page:
        # Gruppierung nach 'locname' und 'timediffofforecast', Berechnung der Mittelwerte für die gewünschten Spalten
        dfDataGroupedTimediffOnly = filtered_df.groupby(['timediffofforecast']).agg({
            'cloud_cover_diff': 'mean',
            'cloud_cover_low_diff': 'mean',
            'cloud_cover_mid_diff': 'mean',
            'cloud_cover_high_diff': 'mean'
        }).reset_index()
        # Hier wird davon ausgegangen, dass du bereits ein dfGrouped hast oder es vorher lädst
        signifikanztest_page(dfDataGroupedTimediffOnly)

    # Seite 4: Tabellenansicht anzeigen
    if 'Signifikanztest II -' in page:
    #if page == 'Signifikanztest II':
        # Datenvorbereitung
        df = dfDataGrouped  # Angenommen, dfDataGrouped enthält alle Daten


        # Funktion zur Klassifizierung der Zeitdifferenzen in 24h-Klassen
        def classify_timediff(timediff):
            return (timediff // 24) * 24


        # Zeitdifferenzen klassifizieren
        df['timediff_class'] = df['timediffofforecast'].apply(classify_timediff)

        # t-Test für jede Messreihe und jeden Ort berechnen
        results = {}

        for loc in selected_locations:
            df_loc = df[df['locname'] == loc]

            # Resultate für diesen Ort speichern
            results[loc] = {}

            for measurement in selected_measurements:
                # Resultat-Matrix für diese Messreihe
                classes = sorted(df_loc['timediff_class'].unique())
                matrix = pd.DataFrame(np.nan, index=classes, columns=classes)

                # t-Test für jede Paarung von 24h-Klassen
                for i, class1 in enumerate(classes):
                    for j, class2 in enumerate(classes):
                        if i < j:
                            group1 = df_loc[df_loc['timediff_class'] == class1][measurement]
                            group2 = df_loc[df_loc['timediff_class'] == class2][measurement]

                            # t-Test durchführen
                            t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
                            print(f" ***  t={t_stat:.2f} <br> p={p_value:.3f}")

                            # Ergebnis in der Matrix speichern
                            #matrix.loc[class1, class2] = f"t={t_stat:.2f}, p={p_value:.3f}"
                            matrix.loc[class1, class2] = f"t={t_stat:.2f}<br>p={p_value:.3f}"

                results[loc][measurement] = matrix

        # Ergebnisse anzeigen
        st.write("Ergebnisse des t-Tests für jede Messreihe und jeden Ort:")
        st.markdown("""
        ### Ergebnisse des **t-Tests** für jede Messreihe und jeden Ort:
        """)

        for loc in selected_locations:
                st.subheader(f"Ort: {loc}")
                for measurement in selected_measurements:
                    st.write(f"Messreihe: {measurement}")
                    #st.dataframe(results[loc][measurement])
                    # Konvertiere DataFrame in HTML mit Zeilenumbrüchen
                    html_table = results[loc][measurement].to_html(escape=False)
                    st.write(html_table, unsafe_allow_html=True)




    # Seite 5: Tabellenansicht anzeigen
    elif page == "Tabellen":
        st.subheader("Datenübersicht")
        st.dataframe(dfDataGrouped[dfDataGrouped['locname'].isin(selected_locations)])
else:
    st.warning("Bitte laden Sie eine CSV-Datei hoch.")