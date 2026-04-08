/*
    Model.java
    Interfaz DESCRIPTORA de los valores NECESARIOS para comprobar que un gen o
    una parte del mismo sean validos

    Copyright (C) 2016.
    Morales Yonathan (yonathan.morales@unet.edu.ve), Jose Lopez (jlopez@unet.edu.ve).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>

*/

package gene.feature;

/**
 * Interfaz DESCRIPTORA de los valores NECESARIOS para comprobar que un gen o
 * una parte del mismo sean validos
 */
public interface Model {
    //---------------------------Static Constants-------------------------------
    // <editor-fold desc="Static Constants">
    //-------- tamaño de un entron y un exon
    public final int maxIntron = 5000;
    public final int minIntron = 64;
    public final int maxExon = 1000;
    public final int minExon = 100;
    public final int minUTR5p = 30;
    public final int maxUTR5p = 1000;
    public final int minUTR3p = 30;
    public final int maxUTR3p = 1000;
    public final int limInfRegionPromo = 1000;
    public final int limSupRegionPromo = 800;
    //---------------------------------------
    public static final String ATG = "atg";
    public static final String GT = "gt";
    public static final String AG = "ag";
    //---------------------------------------
    public static final int TAA = 0;
    public static final int TAG = 1;
    public static final int TGA = 2;
    public static final String[] stops = {"taa", "tag", "tga"};
    //---------------------------------------
    public static final String allowCharacters = "agtc";
    //  </editor-fold>
}
