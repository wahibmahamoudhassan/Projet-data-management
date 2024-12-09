import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import cufflinks as cf
from PIL import Image
from streamlit_option_menu import option_menu
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from wordcloud import WordCloud

# Configuration de la barre latérale
with st.sidebar:
    selection = option_menu(
        "Menu",
        ["Contexte du projet", "Étude du jeu de données", "Dashboard de vente", "Text Mining", "Machine Learning"],
        icons=["house", "graph-up", "bar-chart", "chart-text", "robot"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

# Fonction pour la page de contexte du projet
def project_context():
    st.title("Contexte du projet")
    st.image("image_vente.jpeg", width=400)
    st.markdown("""
    Cette application vise à analyser un jeu de données de vente, à prétraiter les données, à créer un dashboard de vente,
    à effectuer du text mining et à appliquer des techniques de machine learning pour la régression.

    **Objectifs :**
    - Analyser les tendances des ventes.
    - Identifier les segments de clientèle les plus rentables.
    - Effectuer une analyse textuelle pour extraire des informations clés.
    - Appliquer des modèles de machine learning pour améliorer les prévisions des ventes.
    """)
    st.markdown("[Source des données](https://www.kaggle.com/datasets/abhishekrp1517/sales-data-for-economic-data-analysis/data)")
    voir_contact = st.checkbox("Réalisé par :")
    if voir_contact:
        st.markdown("""
                    - Wahib MAHMOUD HASSAN

                    - Abdourahman KARIEH DINI

                    - MAMADOU DIALLO
                    """)
# Fonction pour la page d'étude du jeu de données
def data_study():
    st.title("Étude du jeu de données")
    st.header("Description du jeu de données")
    voir_description = st.checkbox("**Description des colonnes de l'ensemble de données**")
    if voir_description:
        st.markdown("""
                    L'ensemble de données contient des informations sur les transactions de vente, incluant des variables démographiques,
    des données produits et des chiffres financiers. Voici une liste des colonnes disponibles :

    - **Year** : Année de la transaction.
    - **Month** : Mois de la transaction.
    - **Customer Age** : Âge du client au moment de la transaction.
    - **Customer Gender** : Sexe du client.
    - **Country** : Pays où la transaction a eu lieu.
    - **State** : État spécifique.
    - **Product Category** : Grande catégorie du produit.
    - **Sub Category** : Sous-catégorie précise.
    - **Quantity** : Quantité de produits vendus.
    - **Unit Cost** : Coût de production ou d'acquisition par unité.
    - **Unit Price** : Prix de vente par unité.
    - **Cost** : Coût total des produits vendus.
    - **Revenue** : Chiffre d'affaires total.
    """)

    charge_donnee = st.file_uploader("Choisissez un fichier CSV", type="csv")

    if charge_donnee is not None:
        try:
            if charge_donnee.name.endswith("csv"):
                data = pd.read_csv(charge_donnee, delimiter=";")
            elif charge_donnee.name.endswith("txt"):
                data = pd.read_csv(charge_donnee, delimiter="\t")
            elif charge_donnee.name.endswith(("xlsx", "xls")):
                data = pd.read_excel(charge_donnee)
            elif charge_donnee.name.endswith("json"):
                data = pd.read_json(charge_donnee)
            else:
                st.error("Type de fichier non pris en charge.")
                return
            st.success("Fichier chargé avec succès !")

            tabs = st.tabs(["Aperçu de données", "Prétraitement des données", "Statistiques descriptives"])

            with tabs[0]:
                st.dataframe(data.head())
                st.write(f"Nombre de lignes : {data.shape[0]}")
                st.write(f"Nombre de colonnes : {data.shape[1]}")
                st.write(f"Types de colonnes :")
                st.write(data.dtypes.value_counts())

                quant_vars = ['Year', 'Customer Age', 'Quantity', 'Unit Cost', 'Unit Price', 'Cost', 'Revenue']
                qual_vars = ['Date', 'Month', 'Customer Gender', 'Country', 'State', 'Product Category', 'Sub Category']

                st.write(f"**Variables quantitatives :** {len(quant_vars)}")
                st.write(", ".join(quant_vars))
                st.write(f"**Variables qualitatives :** {len(qual_vars)}")
                st.write(", ".join(qual_vars))

            with tabs[1]:
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data["Date"], errors="coerce")

                data['Year'] = data['Year'].fillna(0).astype(int)

                mois_mapping = {
                    "january": 1, "february": 2, "march": 3, "april": 4,
                    "may": 5, "june": 6, "july": 7, "august": 8,
                    "september": 9, "october": 10, "november": 11, "december": 12
                }
                data['Month'] = data['Month'].astype(str).str.lower().map(mois_mapping).fillna(0).astype(int)

                # Nettoyage des valeurs incorrectes dans la colonne "Month"
                data['Month'] = data['Month'].apply(lambda x: x if 1 <= x <= 12 else np.nan)
                data.dropna(subset=['Month'], inplace=True)

                data['Semestre'] = data['Date'].apply(lambda x: 1 if x.month <= 7 else 2)

                data.drop(columns=["Column1", "index"], inplace=True, errors="ignore")

                data.rename(columns={'Cost': 'Cout_tot', 'Revenue': 'Chiffre_affaires'}, inplace=True)

                data['Benefice'] = data['Chiffre_affaires'] - data['Cout_tot']

                st.subheader("Résumé des étapes de prétraitement")
                resume_pretraitement = {
                    "Description": [
                        "Colonne 'Date' convertie au format datetime.",
                        "Noms des mois convertis en valeurs numériques.",
                        "Ajout d'une colonne indiquant le semestre de la vente.",
                        "Colonnes inutiles supprimées pour simplifier l'analyse.",
                        "Colonnes renommées pour plus de clarté.",
                        "Création d'une nouvelle colonne calculant le bénéfice."
                    ],
                    "Code utilisé": [
                        "`data['Date'] = pd.to_datetime(data['Date'], errors='coerce')`",
                        "`data['Month'] = data['Month'].astype(str).str.lower().map(mois_mapping)`",
                        "`data['Semestre'] = data['Date'].apply(lambda x: 1 if x.month <= 7 else 2)`",
                        "`data.drop(columns=['Column1'], inplace=True, errors='ignore')`",
                        "`data.rename(columns={'Cost': 'Cout_tot', 'Revenue': 'Chiffre_affaires'})`",
                        "`data['Benefice'] = data['Chiffre_affaires'] - data['Cout_tot']`"
                    ]
                }
                resume_df = pd.DataFrame(resume_pretraitement)
                st.table(resume_df)
                st.success("Données prétraitées avec succès !")

                quant_vars = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                qual_vars = data.select_dtypes(include=['object']).columns.tolist()

                st.write(f"**Variables quantitatives :** {len(quant_vars)}")
                st.write(", ".join(quant_vars))
                st.write(f"**Variables qualitatives :** {len(qual_vars)}")
                st.write(", ".join(qual_vars))
                st.dataframe(data.head())

            with tabs[2]:
                st.header("Statistiques descriptives")

                var_type = st.selectbox("Choisissez le type de variable", ["Quantitatives", "Qualitatives"])

                if var_type == "Quantitatives":
                    selected_quant_vars = st.multiselect(
                        "Choisissez les variables quantitatives",
                        quant_vars
                    )

                    quant_vis_options = st.selectbox(
                        "Choisissez le type de visualisation pour les variables quantitatives",
                        ["Histogramme", "Box Plot", "Scatter Plot", "Line Plot"]
                    )

                    if selected_quant_vars and quant_vis_options:
                        st.header("Visualisations des variables quantitatives")
                        for var in selected_quant_vars:
                            if quant_vis_options == "Histogramme":
                                st.subheader(f"Histogramme de {var}")
                                fig = px.histogram(data, x=var, title=f"Histogramme de {var}")
                                st.plotly_chart(fig)

                            elif quant_vis_options == "Box Plot":
                                st.subheader(f"Box Plot de {var}")
                                fig = px.box(data, y=var, title=f"Box Plot de {var}")
                                st.plotly_chart(fig)

                            elif quant_vis_options == "Scatter Plot":
                                st.subheader(f"Scatter Plot de {var}")
                                fig = px.scatter(data, x=data.index, y=var, title=f"Scatter Plot de {var}")
                                st.plotly_chart(fig)

                            elif quant_vis_options == "Line Plot":
                                st.subheader(f"Line Plot de {var}")
                                fig = px.line(data, x=data.index, y=var, title=f"Line Plot de {var}")
                                st.plotly_chart(fig)

                elif var_type == "Qualitatives":
                    selected_qual_vars = st.multiselect(
                        "Choisissez les variables qualitatives",
                        qual_vars
                    )

                    qual_vis_options = st.selectbox(
                        "Choisissez le type de visualisation pour les variables qualitatives",
                        ["Bar Chart", "Pie Chart", "Word Cloud"]
                    )

                    if selected_qual_vars and qual_vis_options:
                        st.header("Visualisations des variables qualitatives")
                        for var in selected_qual_vars:
                            if qual_vis_options == "Bar Chart":
                                st.subheader(f"Bar Chart de {var}")
                                fig = px.bar(data[var].value_counts(), title=f"Bar Chart de {var}")
                                st.plotly_chart(fig)

                            elif qual_vis_options == "Pie Chart":
                                st.subheader(f"Pie Chart de {var}")
                                fig = px.pie(data, names=var, title=f"Pie Chart de {var}")
                                st.plotly_chart(fig)

                            elif qual_vis_options == "Word Cloud":
                                st.subheader(f"Word Cloud de {var}")
                                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data[var].astype(str)))
                                plt.figure(figsize=(10, 5))
                                plt.imshow(wordcloud, interpolation='bilinear')
                                plt.axis('off')
                                st.pyplot(plt)

                stats = data[['Customer Age', 'Cout_tot', 'Chiffre_affaires', 'Benefice']].describe()
                st.write("Résume de statistique descriptives des variables quantitatives")
                stats_transposed = stats.T
                st.write(stats_transposed)

                corr_df = data[['Customer Age', 'Cout_tot', 'Chiffre_affaires', 'Benefice']]
                corr_matrix = corr_df.corr()
                st.markdown("**Heatmap des Variables Numériques**")
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)
                st.pyplot(plt)

                # Sauvegarder les données prétraitées pour le tableau de bord
                st.session_state.preprocessed_data = data

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")

def tableau_bord_vente():
    st.title("Dashboard de vente 📊")

    if 'preprocessed_data' in st.session_state:
        data = st.session_state.preprocessed_data

        st.subheader("Aperçu des données")
        st.dataframe(data.head())

        st.sidebar.header("Filtres")
        pays = st.sidebar.multiselect("Filtrer par pays", options=data["Country"].unique(), default=data["Country"].unique())
        categories = st.sidebar.multiselect("Filtrer par catégorie de produit", options=data["Product Category"].unique(), default=data["Product Category"].unique())

        data_filtre = data[(data["Country"].isin(pays)) & (data["Product Category"].isin(categories))]

        st.subheader("Indicateurs clés de performance (KPI)")
        total_ventes = data_filtre["Chiffre_affaires"].sum()
        total_benefice = data_filtre["Benefice"].sum()
        moyenne_prix_unitaire = data_filtre["Unit Price"].mean()
        nombre_transactions = len(data_filtre)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chiffre d'affaires total (€)", f"{total_ventes:,.2f}")
        col2.metric("Bénéfice total (€)", f"{total_benefice:,.2f}")
        col3.metric("Prix moyen (€)", f"{moyenne_prix_unitaire:,.2f}")
        col4.metric("Transactions totales", nombre_transactions)

        st.subheader("Tendances mensuelles des ventes")
        data_filtre["Month"] = pd.to_datetime(data_filtre["Month"], format="%m").dt.strftime("%B")
        ventes_par_mois = data_filtre.groupby("Month")["Chiffre_affaires"].sum().reset_index()
        fig1 = px.bar(
            ventes_par_mois,
            x="Month",
            y="Chiffre_affaires",
            title="Chiffre d'affaires par mois",
            labels={"Chiffre_affaires": "Chiffre d'affaires (€)", "Month": "Mois"},
            color="Chiffre_affaires",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig1)

        st.subheader("Répartition des ventes par catégorie de produit")
        ventes_par_categorie = data_filtre.groupby("Product Category")["Chiffre_affaires"].sum().reset_index()
        fig2 = px.pie(
            ventes_par_categorie,
            names="Product Category",
            values="Chiffre_affaires",
            title="Répartition des ventes par catégorie"
        )
        st.plotly_chart(fig2)

        st.subheader("Répartition géographique des ventes")
        ventes_par_pays = data_filtre.groupby("State")["Chiffre_affaires"].sum().reset_index()
        fig3 = px.bar(
            ventes_par_pays,
            x="State",
            y="Chiffre_affaires",
            title="Chiffre d'affaires par région",
            labels={"Chiffre_affaires": "Chiffre d'affaires (€)", "State": "Région"},
            color="Chiffre_affaires",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig3)

        st.subheader("Analyse des bénéfices par sous-catégorie")
        benefice_par_sous_categorie = data_filtre.groupby("Sub Category")["Benefice"].sum().reset_index()
        fig4 = px.bar(
            benefice_par_sous_categorie,
            x="Sub Category",
            y="Benefice",
            title="Bénéfices par sous-catégorie",
            labels={"Benefice": "Bénéfices (€)", "Sub Category": "Sous-catégorie"},
            color="Benefice",
            color_continuous_scale="Sunset"
        )
        st.plotly_chart(fig4)

        # Ajout des nouvelles visualisations
        st.subheader("Bénéfices, Coûts et Revenu Total par Mois et Année")
        df_grouped = data_filtre.groupby(['Year', 'Month']).agg({
            'Benefice': 'sum',
            'Cout_tot': 'sum',
            'Chiffre_affaires': 'sum'
        }).reset_index()

        # Créer une figure avec deux sous-graphiques pour les bénéfices et les coûts
        fig = make_subplots(
            rows=1, cols=2,  # 1 ligne, 2 colonnes (pour les bénéfices et les coûts)
            subplot_titles=['Bénéfice par Mois et Année', 'Coût et Revenu Total par Mois et Année']
        )

        # Définir les couleurs pour chaque année pour les barres (bénéfices et coûts)
        bar_colors = {
            2015: 'skyblue',  # Couleur pour 2015
            2016: 'lightgreen'  # Couleur pour 2016
        }

        # Définir les couleurs pour les courbes de revenu (RT)
        line_colors = {
            2015: 'orange',  # Couleur pour Revenu 2015
            2016: 'red'  # Couleur pour Revenu 2016
        }

        # Ajouter les barres pour les bénéfices
        for year in df_grouped['Year'].unique():
            filtered_df = df_grouped[df_grouped['Year'] == year]

            # Ajouter un diagramme de bâtons pour les bénéfices
            fig.add_trace(
                go.Bar(
                    x=filtered_df['Month'],  # Mois
                    y=filtered_df['Benefice'],  # Bénéfice
                    name=f'Bénéfice {year}',  # Nom de la trace (légende)
                    marker=dict(color=bar_colors.get(year, 'gray'))  # Appliquer la couleur correspondante
                ),
                row=1, col=1  
            )

        # Ajouter les barres pour les coûts et la courbe du revenu total avec couleurs distinctes
        for year in df_grouped['Year'].unique():
            filtered_df = df_grouped[df_grouped['Year'] == year]

            # Ajouter un diagramme de bâtons pour les coûts
            fig.add_trace(
                go.Bar(
                    x=filtered_df['Month'],  # Mois
                    y=filtered_df['Cout_tot'],  # Coût
                    name=f'Coût {year}',  # Nom de la trace (légende)
                    marker=dict(color=bar_colors.get(year, 'red'))  # Appliquer la couleur correspondante
                ),
                row=1, col=2  # Placer dans le second sous-graphe (Coûts)
            )

            # Ajouter la courbe pour le Revenu Total avec couleur distincte
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['Month'],  # Mois
                    y=filtered_df['Chiffre_affaires'],  # Revenu Total (RT)
                    mode='lines',  # Ligne continue
                    name=f'Revenu Total {year}',  # Nom de la trace (légende)
                    line=dict(shape='linear', color=line_colors.get(year, 'gray'))  # Couleur distincte pour chaque année
                ),
                row=1, col=2  # Placer dans le graphique des coûts
            )

        # Mise à jour de la mise en page
        fig.update_layout(
            title='Bénéfices, Coûts et Revenu Total par Mois et Année',
            xaxis_title='Mois',
            yaxis_title='Valeur',
            barmode='group',  # Barres groupées pour chaque mois
            height=500,  # Hauteur de la figure
            width=1000,  # Largeur de la figure
            showlegend=True  # Afficher la légende
        )

        # Afficher la figure
        st.plotly_chart(fig)

        st.subheader("Bénéfices par Mois et par Pays")
        # Calculer les bénéfices totaux par mois et par pays
        benefits_by_month_country = data_filtre.groupby(['Month', 'Country'])['Benefice'].sum().reset_index()

        # Créer un graphique de type courbe
        fig = px.line(
            benefits_by_month_country,
            x='Month',  # Axe horizontal : Mois
            y='Benefice',  # Axe vertical : Bénéfices
            color='Country',  # Différenciation par pays
            title='Bénéfices par Mois et par Pays',
            labels={'Benefice': 'Bénéfices', 'Month': 'Mois', 'Country': 'Pays'},
            line_shape='linear',  # Forme des courbes
            color_discrete_sequence=px.colors.qualitative.Set2  # Palette de couleurs
        )

        # Personnalisation de l'apparence
        fig.update_layout(
            xaxis=dict(title='Mois', tickmode='linear', tick0=1, dtick=1),  # Mois affichés comme 1, 2, 3...
            yaxis=dict(title='Bénéfices'),
            legend_title='Pays',
            template='plotly_white',  # Style épuré
            height=600,  # Hauteur de la figure
            width=900  # Largeur de la figure
        )

        # Afficher la figure
        st.plotly_chart(fig)

        st.subheader("Sous-catégories les plus vendues (Quantité)")
        # Agréger les données par sous-catégorie
        sub_cat_sales = data_filtre.groupby('Sub Category')['Quantity'].sum().sort_values(ascending=False)

        # Visualisation des sous-catégories les plus vendues
        plt.figure(figsize=(12, 8))
        sns.barplot(x=sub_cat_sales.values, y=sub_cat_sales.index, palette='viridis')

        # Titres et étiquettes
        plt.title('Sous-catégories les plus vendues (Quantité)', fontsize=16)
        plt.xlabel('Quantité totale vendue', fontsize=12)
        plt.ylabel('Sous-catégories', fontsize=12)

        st.pyplot(plt)

    else:
        st.info("Veuillez d'abord prétraiter les données dans la section 'Étude du jeu de données'.")

if selection == "Contexte du projet":
    project_context()
elif selection == "Étude du jeu de données":
    data_study()
elif selection == "Dashboard de vente":
    tableau_bord_vente()
