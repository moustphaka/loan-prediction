from src.train import train_model

def main():
    print("MAIN.PY EST EXECUTÉ")
    print("Démarrage du programme...")

    try:
        model = train_model()
        print("Entraînement terminé avec succès.")

        if model is None:
            print("Attention : train_model() ne retourne aucun modèle.")
        else:
            print("Modèle entraîné et récupéré.")

    except Exception as e:
        print("❌ Erreur pendant l'exécution :")
        print(e)

    print("🏁 Fin du programme.")

if __name__ == "__main__":
    main()