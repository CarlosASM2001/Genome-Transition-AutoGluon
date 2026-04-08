% metodo para escribir en un archivo desde progol
% Benjamin Johnston "http://objectmix.com/prolog/183044-%5Bswi-prolog%5D-read-write-file-how.html"

:- discontiguous buscar/4. 

write_triple(File, A, B, C) :-
open(File, write, Stream),
write(Stream, (A,B,C)),
nl(Stream),
close(Stream).

write_double(File, A, B) :-
open(File, write, Stream),
write(Stream, (A,B)),
nl(Stream),
close(Stream).


% Posibles Patrones dentro las secuencias.
senal('atg').
senal('gt').
senal('ag').
senal('taa').
senal('tga').
senal('tag').

% Tipos basicos
bases([]).
bases([B|Bss]) :- base(B), bases(Bss).


	% Solo hay cuatro posibles bases
%base(p). % caracter marcador
base(a).
base(c).
base(g).
base(t).

	% Valida si la secuencia [B|Bss] se inicia con alguna de las bases
 
b(a, [a|A], A).
b(c, [c|A], A).
b(g, [g|A], A).
b(t, [t|A], A).

	%  y es una base pirimidina
   
b(y, [c|A], A).
b(y, [t|A], A).

	%  r es una base purina
 
b(r, [a|A], A).
b(r, [g|A], A). 

	%  b no es una adenina

b(b, [g|A], A).
b(b, [c|A], A).
b(b, [t|A], A).

	%  d no es una citosina

b(d, [a|A], A).
b(d, [g|A], A).
b(d, [t|A], A).

	%  h no es una guanina

b(h, [a|A], A).
b(h, [c|A], A).
b(h, [t|A], A).

	%  v no es una timina

b(v, [a|A], A).
b(v, [c|A], A).
b(v, [g|A], A).

	%  n es cualquier base

b(n, [a|A], A).
b(n, [c|A], A).
b(n, [g|A], A).
b(n, [t|A], A).

% Posibles polimeros a analizar.
poli('a').
poli('aa').
poli('aaa').
poli('aaaa').
poli('aaaaa').
poli('aaaaaa').
poli('aaaaaaa').
poli('gc').

% Posibles direcciones de estudio
direccion('izq').
direccion('der').

% Patron se corresponde con señal

patron(Senal, Patron):-
	Senal == atg -> Patron = [a,t,g].

patron(Senal, Patron):-
	Senal == gt -> Patron = [g,t].

patron(Senal, Patron):-
	Senal == ag -> Patron = [a,g].

patron(Senal, Patron):-
	Senal == taa -> Patron = [t,a,a].

patron(Senal, Patron):-
	Senal == tga -> Patron = [t,g,a].

patron(Senal, Patron):-
	Senal == tag -> Patron = [t,a,g].

zonas(Secuencia, LI, LD, Senal, Posicion, ZI, ZD, D):-
	patron(Senal, Patron),
	senal_es(Secuencia, Posicion, Patron, 0, Patron, I, D),
	( (Posicion=='null'->ZI=I,ZD=D);
        ( reverse(I, NI), length(NI, TI), length(D, TD),
	( ( TI =< LI -> NZI = NI ); posicionar(NI, 0, LI, NZI, _) ),!, reverse(NZI, ZI),
	( ( TD =< LD -> ZD = D ); posicionar(D, 0, LD, ZD, _) ), ! )).

zonas(Secuencia, LI, LD, Senal, Posicion, ZI, ZD):- 
	patron(Senal, Patron),
	senal_es(Secuencia, Posicion, Patron, 0, Patron, I, D),
	( (Posicion=='null'->ZI=I,ZD=D);
        ( reverse(I, NI), length(NI, TI), length(D, TD),
	( ( TI =< LI -> NZI = NI ); posicionar(NI, 0, LI, NZI, _) ),!, reverse(NZI, ZI),
	( ( TD =< LD -> ZD = D ); posicionar(D, 0, LD, ZD, _) ), ! )).

% Busca Patron y divide la secuencia en estudio en dos sectores (Izq, Der).

% senal([], _, 'null', [], []).

% devuelve las zonas de estudio a ambos lados de la señal

senal(Secuencia, Senal, Posicion, ZI, ZD):-
	patron(Senal, Patron),
	senal_es(Secuencia, Posicion, Patron, 0, Patron, ZI, ZD).

senal(Direccion, Secuencia, Senal, Posicion, Zona):-
	patron(Senal, Patron),
	senal_es(Secuencia, Posicion, Patron, 0, Patron, Izq, Der),
	( Direccion == 'izq' -> (Zona = Izq) ; (Zona = Der) ).

% senal_es([], _, _, _, _, 'null', []) :- !, fail .

senal_es(R, N, [], _, Patron, [], R):- 
  length(Patron, L), N is (0 - L).

senal_es([F|R], N, [P|RP], D, Patron, [F|Izq], Der) :-
  F \== P, D == 0,
  senal_es(R, NN, [P|RP], D, Patron, Izq, Der),
  N is NN + 1.

senal_es([F|R], N, [P|_], D, Patron, Izq, Der) :-
  F \== P, D \== 0, posicionar(Patron, 0, D, I, _),
  senal_es([F|R], N, Patron, 0, Patron, NIzq, Der),
  append(I,NIzq,Izq).
 
senal_es([F|R], N, [F|RP], D, Patron, Izq, Der) :-
  ND is D + 1,
  senal_es( R, NN, RP, ND, Patron, Izq, Der),
  N is NN + 1.

% Une dos listas
append([],L,L).
append([H|T],L2,[H|L3]) :- append(T,L2,L3).

%	reverse(?List1, ?List2)
%
%	Is true when the elements of List2 are in reverse order compared to
%	List1.

reverse(Xs, Ys) :-
    reverse(Xs, [], Ys, Ys).

reverse([], Ys, Ys, []).
reverse([X|Xs], Rs, Ys, [_|Bound]) :-
    reverse(Xs, [X|Rs], Ys, Bound).

% Avanza en el recorrido de la secuencia [B|NBss] una vez ha validado que B es un nucleotido.
avanz_pos([B|NBss], NBss) :- base(B).

% Posiciona la busqueda en un punto especifico (N) y devuelve las secuencia a la derecha
posicionar([], _, _, []).
posicionar(_, I, N, _) :- I > N, !, fail.
posicionar(R, I, N, R) :- I == N, !.
posicionar([_|R], I, N, RR ) :- II is I+1, posicionar(R, II, N, RR).

% Posiciona la busqueda en un punto especifico (N) y devuelve las secuencia a la izquierda y derecha
posicionar([], _, _, [], []).
posicionar(_, P, L, _, _) :- P > L, !, fail.
posicionar(R, P, L, [], R) :- P == L, !.
posicionar([F|R], P, L, [F|NR], RR) :- PP is P+1, posicionar(R, PP, L, NR, RR).

% Limites de exploracion para algunos conceptos
limite(100).
limite_pirimidinas(50).
limite_adenil(100).
limite_ramificacion(60).
lim_atg_p_par(30).
umbral_atg_p_par(1).
lim_atg_s_par(30).
umbral_atg_s_par(1).
lim_atg_RYn(40).
umbral_atg_RYn(8).
tipo_analisis(no_inmediato).
dist_max_gt_vec_prec_ag(50).
umbral_gt_seg_ATn(6).
dist_max_ag_vecino(40).
limite_ag_polyAes(40).
umbral_ag_polyAes(0.4).
umbral_agpre_aags(2). 
limite_agpre_aags(40).
limite_poliadeni(30).

% Posibles tipos de transicion
trans('no_trans').
trans('no_cod->cod').
trans('cod->no_cod').

% Analisis de polimeros por proporciones

polimeros(Polimero, Secuencia, Direccion, Senal, Posicion, Proporcion):-  %Cantidad contiene el numero de dimeros aa en en la Secuencia.
	limite(Limite), % Establece la longitud maxima de exploracion de bases de ADN a la derecha o izquierda de una señal caracteristica.
	Polimero == 'a',  %Valida la version del concepto polimeros a emplear; en este caso interesa  analizar el dimero 'aa'.
	senal(Direccion, Secuencia, Senal, Posicion, Zona), length(Zona, TZ), 
	( ( TZ =< Limite -> NLimite = TZ ); NLimite = Limite ),
	buscar(Zona, Polimero, NLimite, Cantidad),!,
	Proporcion is (Cantidad/NLimite).

polimeros(Polimero, Secuencia, Direccion, Senal, Posicion, Proporcion):-  %Cantidad contiene el numero de dimeros aa en en la Secuencia.
	limite(Limite), % Establece la longitud maxima de exploracion de bases de ADN a la derecha o izquierda de una señal caracteristica.
	Polimero == 'gc',  %Valida la version del concepto polimeros a emplear; en este caso interesa  analizar el dimero 'aa'.
	senal(Direccion, Secuencia, Senal, Posicion, Zona), length(Zona, TZ),
	( ( TZ =< Limite -> NLimite = TZ ); NLimite = Limite ),
	buscar(Zona, Polimero, NLimite, Cantidad),!,
	Proporcion is (2*Cantidad/NLimite).

buscar(Secuencia, Polimero, Limite, Cantidad):-   % cantidad contiene el numero de polimeros ryry en en la Secuencia, hasta Limite.
	Polimero == 'a',     		  % En este caso interesa  analizar el dimero 'aa'.
	a(Posicion, Secuencia, Resto), V is Limite - 1, Posicion < V,
	(NLimite is (Limite - (Posicion + 1))),
	( NLimite > 0  -> ( buscar(Resto, Polimero, NLimite, NCantidad), (Cantidad is 1 + NCantidad) ) ; (Cantidad is 1) ).

buscar(Secuencia, Polimero, Limite, Cantidad):-   % cantidad contiene el numero de polimeros ryry en en la Secuencia, hasta Limite.
	Polimero == 'gc',     		  % En este caso interesa  analizar el dimero 'gc'.
	gc(Posicion, Secuencia, Resto), V is Limite - 2, Posicion < V,	  % Busca primer dimero 'aa' y devuelve su posicion. Resto_S no incluye al dimero hallado.
	(NLimite is (Limite - (Posicion + 2))),
	( NLimite > 0  -> ( buscar(Resto, Polimero, NLimite, NCantidad), (Cantidad is 1 + NCantidad) ) ; (Cantidad is 1) ).

buscar(_, _, _, 0).

% Halla una 'a', devuelve su posicion y el resto de la secuencia.

a(_, [], _) :- true,!, fail.

a(0, Secuencia, Resto):-  b(a, Secuencia, Resto).   

a(Posicion, Secuencia, RestoS):-
	avanz_pos(Secuencia, Resto), 
	a(NPosicion, Resto, RestoS), 
	Posicion is (1 + NPosicion).

% Halla un dimero 'gc', devuelve su posicion y el resto de la secuencia.

gc(_, [], _) :- !, fail.

gc(0, Secuencia, Resto):-  b(g, Secuencia, NResto), b(c, NResto, Resto).   

gc(Posicion, Secuencia, Resto):-
	avanz_pos(Secuencia, RestoS), 
	gc(NPosicion, RestoS, Resto), 
	Posicion is (1 + NPosicion).


% Piso y techo establecen limites inferior y superior para la proporcion de un dimero determinado.

% si X es constante y existe un piso, verifica que Y contenga ese valor y compara.
piso(X,Y) :- not(var(X)), not(var(Y)), float(X),  float(Y), X >= Y.

% si X es constante y existe un techo Y, verifica que Y contenga ese valor y compara.
techo(X,Y) :- not(var(X)), not(var(Y)), int(X),  int(Y), X =< Y.

% si X es constante y existe un piso, verifica que Y contenga ese valor y compara.
entre(X,Y) :- not(var(X)), not(var(Y)), int(X),  int(Y), X >= Y.

% si X es constante y existe un techo Y, verifica que Y contenga ese valor y compara.
y(X,Y) :- not(var(X)), not(var(Y)), int(X),  int(Y), X =< Y.










