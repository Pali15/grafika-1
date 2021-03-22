//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Bakó Pál Ignác
// Neptun : I31TDE
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec2 vertexUV;

	out vec2 textCoord;

	void main() {
		textCoord=vertexUV;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform sampler2D textureUnit;

	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 fragmentColor;

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
		fragmentColor = texture(textureUnit, texCoord);
	}
)";




GPUProgram gpuProgram; // vertex and fragment shaders


vec2 ConvertToVec2(vec3 a) {
	return vec2(a.x / a.z, a.y / a.z);
}

vec3 ConvertToVec3(vec2 a) {
	float divider = sqrt(1 - a.x * a.x - a.y * a.y);
	return vec3(a.x / divider, a.y / divider, 1 / divider);
}


bool TwoSegmentsIntersects(vec2 a_1, vec2 a_2, vec2 b_1, vec2 b_2) {//vizsgálja hogy két szakasz metszi-e egymást
	
	/*
	* elv:
	* Elsõnek megvizsgáljuk hogy az x tengelyen lévõ intervallumaiknak van-e közös része. Ha nincs akkor nem metszehtik egymást, ha van megyünk tovább
	* Felírjuk a két vektorra illesztett egyenes egyenletét és ha azok kiegyenlítik egymást akkor van metszõ pont aminek megvannak a koordinátái
	* Megnézzük, hogy a metszõ pont x koordinátája eleme e a két intervallum közös részének
	* két intervallum közös része [két baloldal közül a nagyobbik; két jobboldal közül a kisebbik]
	*/


	//a bal és jobb oldali pontja
	float min_a = a_1.x < a_2.x ? a_1.x : a_2.x;
	float max_a = a_1.x > a_2.x ? a_1.x : a_2.x;
	//b bal és jobb oldali pontja
	float min_b = b_1.x < b_2.x ? b_1.x : b_2.x;
	float max_b = b_1.x > b_2.x ? b_1.x : b_2.x;

	
	if (max_a < min_b) return false; //az intervallumok x tengely mentén nem metszik egymást, tehát nem metszhetik egymást a vektorok

	//egyenes egyenlete: A1*x + b1 = y

	//meredekségek
	float A1 = 0;
	float A2 = 0;

	if (a_1.x - a_2.x != 0)
		A1 = (a_1.y - a_2.y) / (a_1.x - a_2.x);

	if (b_1.x - a_2.x != 0)
		A2 = (b_1.y - b_2.y) / (b_1.x - b_2.x);
	
	//eltolás y tengelyen
	float b1 = a_1.y - A1 * a_1.x;
	float b2 = b_1.y - A2 * b_1.x;

	//A pont ahol metszik egymást legyen P(X,Y)
	//Akkor metszik egymást, ha A1*x + b1=A2*x + b2
	float X = (b2 - b1) / (A1 - A2);

	//
	float greatest_min = min_a > min_b ? min_a : min_b;
	float lowest_max = max_a < max_b ? max_a : max_b;

	if (X<greatest_min || X>lowest_max)
		return false;
	else
		return true;
}

int Intersects(vec2 arr[122]) {//visszaadja hogy hány metszõ él van
	int sum = 0;
	for (int i = 0; i < 120; i += 2) {
		if (i > 1) {
			for (int j = 0; j < 120; j += 2) {
				if (j != i) {
					if (TwoSegmentsIntersects(arr[i], arr[i + 1], arr[j], arr[j + 1]))
						sum++;
				}
			}
		}
	}

	return sum;
}

void SetNeighbours(vec2 old[122], vec2 next[122]) {
	for (int i = 0; i < 122; i++) {
		old[i] = next[i];
	}
}


void SetVertices(vec2 old[50], vec2 next[50]) {
	for (int i = 0; i < 50; i++) {
		old[i] = next[i];
	}
}

class Graph {

public:
	vec2 circle[100];
	vec2 UV[4];

	vec3 vertices3D[50];//50pont a hiperbolikos síkon
	vec2 vertices[50];//50 pont a gráfban

	int indexes[122];//122 random index a szomszédokból
	vec2 neighbors[122];//61 él kell->122pont a 61 élhez

public:

	//true-val tér vissza ha egy szomszédság már létezik
	bool neighborAlreadyExists(int arr[122], int a, int b, int elements) {
		for (int i = 0; i < (elements - 2); i += 2) {
			if ((a == arr[i] && b == arr[i + 1]) || (b == arr[i] && a == arr[i + 1])) {
				return true;
			}
		}
		return false;
	}

	//feltölti az indexes és a vertices tömböket
	void InitVertices() {

		//indexes
		for (int i = 0; i < 122; i += 2) {


			int r = rand() % 50;//random index1
			int r2 = rand() % 50;//random index2

			while (neighborAlreadyExists(indexes, r, r2, i)) {//if they are following each other somewhere int the array than that neighborhood is already existing
				r = rand() % 50;
				r2 = rand() % 50;
			}

			indexes[i] = r;
			indexes[i + 1] = r2;
		}

		//vertices
		for (int i = 0; i < 50; i++) {//creating random 50 points for vertices


			float x = ((float(rand()) / float(RAND_MAX)) * (2)) - 1;
			float y = ((float(rand()) / float(RAND_MAX)) * (2)) - 1;
			float z = sqrt(1 + x * x + y * y);

			vertices3D[i].x = x;
			vertices3D[i].y = y;
			vertices3D[i].z = z;

			vertices[i] = ConvertToVec2(vertices3D[i]);		
		}
	}

	void InitNeighbors() {
		for (int i = 0; i < 122; i++) {//Init neighbors
			neighbors[i] = vertices[indexes[i]];
			//printf("%f %f -- %f %f\n", neighbors[i].x, vertices[indexes[i]].x, neighbors[i].y, vertices[indexes[i]].y);
		}
	}

	void Circle(int index) {//kirajzol egy kört az átvett indexü pont középponttal
		for (int i = 0; i < 100; i++) {

			float fi = i * 2 * M_PI / 100;

			circle[i] = (vec2(cosf(fi) * 0.04f, sinf(fi) * 0.04f)+vec2(vertices3D[index].x, vertices3D[index].y));

			float divider = sqrt(1 + circle[i].x * circle[i].x + circle[i].y * circle[i].y);

			circle[i] = circle[i] / divider;
		}

		UV[0] = circle[24];
		UV[1] = circle[49];
		UV[2] = circle[74];
		UV[3] = circle[99];

		glBindBuffer(GL_ARRAY_BUFFER, vbo2[1]);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(graph.UV),  // # bytes
			graph.UV,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(1);  // AttribArray 0
		glVertexAttribPointer(0,      // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); // stride, offset: tightly packed*/
	}

	
	void PrintVertices() {
		for (int i = 0; i < 50; i++) {
			printf("%d.: %f %f\n", (i+1), vertices[i].x, vertices[i].y);
		}
	}

	void PrintNeighbors() {
		int j = 1;
		for (int i = 0; i < 119; i += 2) {
			//printf("%d.: %f %f --> %f %f\n", j, neighbors[i].x, neighbors[i].y, neighbors[i + 1].x, neighbors[i + 1].y);
			j++;
		}
	}

	void Printindexes() {
		int j = 1;
		for (int i = 0; i < 119; i += 2) {
			printf("%d.: %d --> %d\n", j, indexes[i], indexes[i+1]);
			j++;
		}
	}


	void GetBestOption() {//Getting the best option by intersecting segments
		InitVertices();
		InitNeighbors();
		int min_intersects = 10000;
	
		vec2 best_vertices[50];
		vec2 best_neighbors[122];

		for (int j = 0; j < 100; j++) {
			if (min_intersects > Intersects(neighbors)) {
				min_intersects = Intersects(neighbors);
				SetNeighbours(best_neighbors, neighbors);
			}
			InitNeighbors();
		}
		
		SetNeighbours(neighbors, best_neighbors);
	}

	Graph() {
		InitVertices();
		InitNeighbors();
		
/*		PrintVertices();
		PrintNeighbors();
		Printindexes();*/
	}
};

unsigned int vao;	// virtual world on the GPU
unsigned int vao1;
unsigned int vao2;//circle

unsigned int vbo;
unsigned int vbo1;
unsigned int vbo2[2];
Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glGenVertexArrays(1, &vao1);
	glGenVertexArrays(1, &vao2);


	glBindVertexArray(vao2);
	glGenBuffers(2, vbo2);

	glBindVertexArray(vao1);
	glGenBuffers(1, &vbo1);	


	glBindVertexArray(vao);		
	glGenBuffers(1, &vbo);	
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	
	//saving vertices
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(graph.vertices),  // # bytes
		graph.vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); // stride, offset: tightly packed

	//saving Neighbors
	glBindVertexArray(vao1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo1);



	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(graph.neighbors),  // # bytes
		graph.neighbors,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,      // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); // stride, offset: tightly packed*/



	glBindVertexArray(vao2);
	glBindBuffer(GL_ARRAY_BUFFER, vbo2[0]);

	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(graph.circle),  // # bytes
		graph.circle,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,      // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); // stride, offset: tightly packed*/

	

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}


void DrawCircles() {
	glBindVertexArray(vao2);


	for (int i = 0; i < 50; i++) {
		graph.Circle(i);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(graph.circle),  // # bytes
			graph.circle,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,      // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); // stride, offset: tightly packed*/

		gpuProgram.setUniform(texture, "textureUnit");

		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 100 /*# Elements*/);
	}
	
}

void DrawGraph() {

	glBindVertexArray(vao);
	//saving vertices
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(graph.vertices),  // # bytes
		graph.vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); // stride, offset: tightly packed
	glDrawArrays(GL_POINTS, 0 /*startIdx*/, 50 /*# Elements*/);

	//saving Neighbors
	glBindVertexArray(vao1);
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(graph.neighbors),  // # bytes
		graph.neighbors,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,      // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); // stride, offset: tightly packed*/
	glDrawArrays(GL_LINES, 0 /*startIdx*/, 122 /*# Elements*/);
}



// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location


	glPointSize((GLfloat)4);

	DrawGraph();
	DrawCircles();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}


float lorentz(vec3 a, vec3 b) {
	return a.x * b.x + a.y * b.y - a.z * b.z;
}

void Moving(vec2 MousePos) {
	vec3 Origo(0, 0, 1);
	vec3 Q = ConvertToVec3(MousePos);//A vektor amivel tolunk

	float dist = acosh(-lorentz(Origo, Q));//Origo és a Q távolsága a hiperbolikus síkon
	
	vec3 v;//irányvektor

	if (dist == 0)//nem osztunk 0-val
		return;

	v = (Q - (Origo * cosh(dist))) / sinh(dist);//irányvektor a hiperbolikus síkon

	//2 pont amire tükrözünk úgy hogy dist(m1, m2)=dist(Origo, Q)/2<--ezért van dist/4 és 3*dist/4 mivel igy 2/4 lesz a távolságuk
	vec3 m1 = (Origo * cosh(dist / 4)) + (v * sinh(dist / 4));//m1 az OriogoQ vektoron 
	vec3 m2 = (Origo * cosh(3*dist / 4)) + (v * sinh(3*dist / 4));//m2 az OriogoQ vektoron

	
	//az összes pontunkat tükrözzük az m1-re aztán az m2-re
	for (int i = 0; i < 50; i++) {
		vec3 t = graph.vertices3D[i];//csak azért hogy kevesebbet kelljen írni a késõbiekben

		float dist1 = acosh(-lorentz(m1, t));//m1 t távolság
		if (dist1 == 0)//0-val nem osztunk
			return;

		vec3 v1 = (m1 - (t * cosh(dist1))) / sinh(dist1);//irányvektor t pontban
		vec3 t1 = (t * cosh(2 * dist1)) + (v1 * sinh(dist1 * 2));//t tükrözve m1-re

		float dist2 = acosh(-lorentz(m2, t1));//t1 m2 távolság
		if (dist2 == 0)//0-val nem osztunk
			return;

		vec3 v2 = (m2 - (t1 * cosh(dist2))) / sinh(dist2);//irányvektor t1 pontban
		vec3 t2 = (t1 * cosh(2 * dist2)) + (v2 * sinh(dist2 * 2));//t1 tükrözve m2-re

		graph.vertices3D[i] = t2;
		graph.vertices[i] = ConvertToVec2(graph.vertices3D[i]);
	}

	graph.InitNeighbors();
}




vec2 Start=(0, 0, 0);
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	
	char* buttonStat;
	switch (state) {
	case GLUT_DOWN:
		if (cX * cX + cY * cY < 1) {
			Start.x = cX;
			Start.y = cY;
		}
		buttonStat = "pressed";
		break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
	glutPostRedisplay();
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (cX * cX + cY * cY >= 1)
		return;

	Moving(vec2(cX - Start.x, cY - Start.y));
	Start.x = cX;
	Start.y = cY;

	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
	glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	
}