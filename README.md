---

# README

Nella repository sono presenti i seguenti script Python:

* `decoder.py`: modello utilizzato per decodificare le rappresentazioni ULIP2 in point cloud.

* `train.py`: codice per il training del decoder. Utilizza la Chamfer Loss e una procedura di subsampling per ridurre l’utilizzo di memoria GPU.

* `dataloader.py`: script per il caricamento del dataset.

* `pc_transformation.py`: a partire da una point cloud e da una coppia di testi che definiscono un polo positivo e uno negativo di una determinata caratteristica:

  1. calcola l’embedding della point cloud tramite un modello ULIP2;
  2. calcola la direzione nello spazio latente definita dai due testi;
  3. se gli embedding testuali sono già presenti, evita di ricaricare il checkpoint ULIP2 (operazione che richiede alcuni minuti);
  4. sposta linearmente l’embedding lungo tale direzione tramite il parametro `ALPHA`;
  5. normalizza nuovamente il punto sull’ipersfera latente;
  6. se è stato scelto un valore `ADAPT_STEPS > 0`, viene inoltre eseguito un fine-tuning del checkpoint del decoder sulla specifica istanza da modificare;
  7. viene generata la point cloud corrispondente al nuovo punto nello spazio latente;
  8. il risultato viene salvato nella stessa cartella della point cloud originale, con un nome che include:
      * valore di `ALPHA`;
      * testi positivo e negativo;
      * numero di `ADAPT_STEPS`.

Nelle cartelle:

* `transform_input_face`
* `transform_input_donkey`
* `transform_input_spot`

sono presenti alcuni esempi di esecuzione dello script `pc_transformation.py`, ottenuti a partire da diverse point cloud iniziali e con differenti valori di `ADAPT_STEPS` e `ALPHA`.
