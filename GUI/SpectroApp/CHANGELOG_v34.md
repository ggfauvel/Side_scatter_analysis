# SpectroApp — v34

Trois fichiers : `core/angles.py`, `app/pages/p1_data.py`,
`app/pages/p6_angular.py`. Remplace la v33.

---

## La faute, et elle est ancienne

Votre `_fiber_angles_to_xyz` :

```python
theta_vals = phi_stored        # les roles de phi et theta sont ECHANGES
phi_vals   = theta_stored
args_min = phi_vals < 0        # quadrant negatif -> azimut miroir
phi_vals[args_min]  *= -1
theta_vals[args_min] = 180.0 - theta_vals[args_min]
theta_vals += -90.0
x = sin(theta_r) * sin(phi_r)
y = cos(phi_r)                 # axe polaire = y
z = cos(theta_r) * sin(phi_r)
```

Rien a voir avec la convention spherique usuelle. **L'axe polaire est y**, et
l'angle a la retrodiffusion (+y) vaut **exactement theta**. Votre fleche verte
selon y etait donc juste depuis le debut.

En v25, quand j'ai ecrit `build_config_from_map`, j'ai calcule et stocke les
coordonnees avec ma propre formule (`x=sinθcosφ, y=sinθsinφ, z=cosθ`).
`fig_sphere` prefere les coordonnees fournies quand il y en a : il utilisait
donc les miennes au lieu d'appeler la votre. C'est l'origine unique de tout ce
que nous avons chasse depuis — et mes « corrections » successives (tourner la
formule en v29, l'ordre des fibres en v32, la fleche en v33) ne faisaient que
deplacer le probleme. J'aurais du lire ce code il y a cinq versions au lieu de
raisonner sur des captures ; j'en suis desole.

## Correction

`xyz_from_angles` **delegue** desormais a `sf._fiber_angles_to_xyz`. Il n'y a
plus qu'une conversion dans toute l'application, la votre. La fleche du laser
reprend ses valeurs d'origine et n'est plus un reglage.

Structure finale, celle que vous decriviez :

| | |
|---|---|
| conversion angles → x, y, z | **unique**, celle du pipeline |
| direction du laser | **unique**, imposee par l'experience |
| ordre des fibres | option, par configuration |
| angle negatif selon le quadrant | option, par configuration |

## Verifie

```
NOVEMBRE  ordre = physical (inversees) | quadrants alpha, delta
          coordonnees identiques a celles du code d'origine : oui

MAI       ordre = extraction (non inversees)
          etiquettes  1-2  (faibles)     : 84°, 79° de la retrodiffusion
          etiquettes 10-12 (brillantes)  : 33°, 27°, 21°
          gradient decroissant depuis la retrodiffusion : oui

les 9 pages s'ouvrent : oui
```

Novembre est reproduit au bit pres, et mai devient coherent par la meme
conversion, sans rien de particulier : seules les fibres sont dans l'autre
ordre.

## A faire apres installation

Revalidez la page 1 : les configurations deja en registre contiennent encore
des coordonnees calculees avec mon ancienne formule, et le registre est
reecrit a chaque validation.
