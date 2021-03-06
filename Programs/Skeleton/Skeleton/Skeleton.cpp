//=============================================================================================
// Mintaprogram: Z?ld h?romsz?g. Ervenyes 2019. osztol.
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
// Nev    : Bak? P?l Ign?c
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
uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;														// copy texture coordinates
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	uniform int isTexture;

	uniform vec3 color;	

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 outColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		if(isTexture==1){
			outColor = texture(textureUnit, texCoord);
		}else{
			outColor = vec4(color, 1);	// computed color is the color of the primitive
		}		
	}
)";




GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	// virtual world on the GPU
unsigned int vao1;
unsigned int vao2;//circle

unsigned int vbo;
unsigned int vbo1;
unsigned int vbo2[3];

vec2 ConvertToVec2(vec3 a) {
	return vec2(a.x / a.z, a.y / a.z);
}

vec3 ConvertToVec3(vec2 a) {
	float divider;
	if ((a.x * a.x + a.y * a.y) >= 1) {
		divider = sqrtf(0.01);
	}
	else {
		divider = sqrtf(1 - a.x * a.x - a.y * a.y);
	}
	return vec3(a.x / divider, a.y / divider, 1 / divider);
}


bool TwoSegmentsIntersects(vec2 a_1, vec2 a_2, vec2 b_1, vec2 b_2) {//vizsg?lja hogy k?t szakasz metszi-e egym?st
	
	/*
	* elv:
	* Els?nek megvizsg?ljuk hogy az x tengelyen l?v? intervallumaiknak van-e k?z?s r?sze. Ha nincs akkor nem metszehtik egym?st, ha van megy?nk tov?bb
	* Fel?rjuk a k?t vektorra illesztett egyenes egyenlet?t ?s ha azok kiegyenl?tik egym?st akkor van metsz? pont aminek megvannak a koordin?t?i
	* Megn?zz?k, hogy a metsz? pont x koordin?t?ja eleme e a k?t intervallum k?z?s r?sz?nek
	* k?t intervallum k?z?s r?sze [k?t baloldal k?z?l a nagyobbik; k?t jobboldal k?z?l a kisebbik]
	*/


	//a bal ?s jobb oldali pontja
	float min_a = a_1.x < a_2.x ? a_1.x : a_2.x;
	float max_a = a_1.x > a_2.x ? a_1.x : a_2.x;
	//b bal ?s jobb oldali pontja
	float min_b = b_1.x < b_2.x ? b_1.x : b_2.x;
	float max_b = b_1.x > b_2.x ? b_1.x : b_2.x;

	
	if (max_a < min_b) return false; //az intervallumok x tengely ment?n nem metszik egym?st, teh?t nem metszhetik egym?st a vektorok

	//egyenes egyenlete: A1*x + b1 = y

	//meredeks?gek
	float A1 = 0;
	float A2 = 0;

	if (a_1.x - a_2.x != 0)
		A1 = (a_1.y - a_2.y) / (a_1.x - a_2.x);

	if (b_1.x - a_2.x != 0)
		A2 = (b_1.y - b_2.y) / (b_1.x - b_2.x);
	
	//eltol?s y tengelyen
	float b1 = a_1.y - A1 * a_1.x;
	float b2 = b_1.y - A2 * b_1.x;

	//A pont ahol metszik egym?st legyen P(X,Y)
	//Akkor metszik egym?st, ha A1*x + b1=A2*x + b2
	float X = (b2 - b1) / (A1 - A2);

	//
	float greatest_min = min_a > min_b ? min_a : min_b;
	float lowest_max = max_a < max_b ? max_a : max_b;

	if (X<greatest_min || X>lowest_max)
		return false;
	else
		return true;
}

int Intersects(int arr[122], vec2 vert[50]) {//visszaadja hogy h?ny metsz? ?l van
	int sum = 0;
	for (int i = 0; i < 120; i += 2) {
		if (i > 1) {
			for (int j = 0; j < 120; j += 2) {
				if (j != i) {
					if (TwoSegmentsIntersects(vert[arr[i]], vert[arr[i+1]], vert[arr[j]], vert[arr[j+1]]))
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

//true-val t?r vissza ha egy szomsz?ds?g m?r l?tezik
bool neighborAlreadyExists(int arr[122], int a, int b, int elements) {
	for (int i = 0; i < (elements - 2); i += 2) {
		if ((a == arr[i] && b == arr[i + 1]) || (b == arr[i] && a == arr[i + 1])) {
			return true;
		}
	}
	return false;
}

void SetVertices(vec2 old[50], vec2 next[50]) {
	for (int i = 0; i < 50; i++) {
		old[i] = next[i];
	}
}

class Graph{

public:
	vec2 circle[100];
	vec2 vert[8];
	Texture* texture[50];

	vec3 vertices3D[50];//50pont a hiperbolikos s?kon
	vec2 vertices[50];//50 pont a gr?fban

	int indexes[122];//122 random index a szomsz?dokb?l
	vec2 neighbors[122];//61 ?l kell->122pont a 61 ?lhez

public:

	void InitIndexes() {//61 cs?cs p?r legener?l?sa
		for (int i = 0; i < 122; i += 2) {


			int r = rand() % 50;//random index1
			int r2 = rand() % 50;//random index2

			while (neighborAlreadyExists(indexes, r, r2, i)) {
				r = rand() % 50;
				r2 = rand() % 50;
			}

			indexes[i] = r;
			indexes[i + 1] = r2;
		}
	}

	void InitVertices() {//cs?csok legenr?l?sa

		for (int i = 0; i < 50; i++) {

			float x = ((float(rand()) / float(RAND_MAX)) * (2)) - 1;
			float y = ((float(rand()) / float(RAND_MAX)) * (2)) - 1;
			float z = sqrt(1 + x * x + y * y);

			vertices3D[i].x = x;
			vertices3D[i].y = y;
			vertices3D[i].z = z;

			vertices[i] = ConvertToVec2(vertices3D[i]);
		}
	}

	void GetOptimalSet() {

		int min_intersects = 10000;
		vec2 optimalVertices[50];
		//vertices
		for (int i = 0; i < 5000; i++) {//100 grafot gener?lunk


			float x = ((float(rand()) / float(RAND_MAX)) * (2)) - 1;
			float y = ((float(rand()) / float(RAND_MAX)) * (2)) - 1;
			float z = sqrt(1 + x * x + y * y);

			vertices3D[i-(50*(i/50))].x = x;
			vertices3D[i - (50 * (i / 50))].y = y;
			vertices3D[i - (50 * (i / 50))].z = z;

			vertices[i - (50 * (i / 50))] = ConvertToVec2(vertices3D[i - (50 * (i / 50))]);

			//100 esetb?l kiv?lasztjuk azt amelyikben a legkevesebb metsz? ?l van
			if ((i - (50 * (i / 50))) == 49) {
				if (Intersects(indexes, vertices) < min_intersects) {
					min_intersects = Intersects(indexes, vertices);
					SetVertices(optimalVertices, vertices);
				}
			}
		}

		SetVertices(vertices, optimalVertices);

		for (int i = 0; i < 50; i++) {
			vertices3D[i] = ConvertToVec3(vertices[i]);
		}

		InitNeighbors();
	}

	void InitNeighbors() {
		for (int i = 0; i < 122; i++) {//Init neighbors
			neighbors[i] = vertices[indexes[i]];
		}
	}

	void Circle(int index) {//kirajzol egy k?rt az ?tvett index? pont k?z?pponttal

		for (int i = 0; i < 100; i++) {

			float fi = i * 2 * M_PI / 100;

			circle[i] = (vec2(cosf(fi) * 0.04f, sinf(fi) * 0.04f)+vec2(vertices3D[index].x, vertices3D[index].y));

			float divider = sqrt(1 + circle[i].x * circle[i].x + circle[i].y * circle[i].y);

			circle[i] = circle[i] / divider;
		}

		vert[0] = circle[0];
		vert[1] = circle[24];
		vert[2] = circle[49];
		vert[3] = circle[74];

		vert[4] = vec2(0, 0);
		vert[5] = vec2(1, 0);
		vert[6] = vec2(1, 1);
		vert[7] = vec2(0, 1);

	}

	void InitTextures() {//felt?li a text?ra t?mb?t
		for (int i = 0; i < 50; i++) {
			int width = 8, height = 8;				// create checkerboard texture procedurally
			std::vector<vec4> image(width * height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					//random sz?n gener?l?sa
					float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
					float g = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
					float b = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
					image[y * width + x] = vec4(r, g, b, 1);
				}
			}

			texture[i] = new Texture(width, height, image);
		}
	}

	void DrawCircles() {
		glBindVertexArray(vao2);
		glBindBuffer(GL_ARRAY_BUFFER, vbo2[0]);

		for (int i = 0; i < 50; i++) {
			//els?nek kirajzoljuk a k?rt
			gpuProgram.setUniform(0, "isTexture");//color sz?n? legyen a pixel
			Circle(i);
			glBufferData(GL_ARRAY_BUFFER,
				sizeof(circle),
				circle,
				GL_STATIC_DRAW);

			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0,
				2, GL_FLOAT, GL_FALSE,
				0, NULL);
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats
			glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 100 /*# Elements*/);

			gpuProgram.setUniform((*texture[i]), "textureUnit");//bet?ltj?k az aktu?lis textureUnit-ot
			gpuProgram.setUniform(1, "isTexture");//mostm?r a text?r?hoz igaz?tja a pixel sz?n?t
			glBufferData(GL_ARRAY_BUFFER,//bet?ltj?ka vert-et a bufferbe
				sizeof(vert),
				vert,
				GL_STATIC_DRAW);
			//bet?ltj?k a vertet
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0,
				2, GL_FLOAT, GL_FALSE,
				0, NULL);
			//bet?ltjuk a cs?cspontokat
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1,
				2, GL_FLOAT, GL_FALSE,
				0, (void*)(4*sizeof(vec2)));//4*vec2 offset-->az utols? 4 elemet t?ltse be

			glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 4 /*# Elements*/);//kirajzoljuk
		}
	}

	void DrawGraph(){

		//saving Neighbors
		glBindVertexArray(vao1);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(neighbors),  // # bytes
			neighbors,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,      // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); // stride, offset: tightly packed*/
		int location = glGetUniformLocation(gpuProgram.getId(), "isTexture");
		glUniform1i(location, 0);

		location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 0.0f); // 3 floats

		glDrawArrays(GL_LINES, 0 /*startIdx*/, 122 /*# Elements*/);
	}

	Graph() {
		InitIndexes();
		InitVertices();
		InitNeighbors();
		InitTextures();
	}
};


Graph* graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glGenVertexArrays(1, &vao1);
	glGenVertexArrays(1, &vao2);


	glBindVertexArray(vao2);
	glGenBuffers(3, vbo2);

	glBindVertexArray(vao1);
	glGenBuffers(1, &vbo1);


	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,      // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); // stride, offset: tightly packed*/

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
	graph = new Graph();
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

	graph->DrawGraph();
	graph->DrawCircles();
	glutSwapBuffers(); // exchange buffers for double buffering
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

	float dist = acosh(-lorentz(Origo, Q));//Origo ?s a Q t?vols?ga a hiperbolikus s?kon
	
	vec3 v;//ir?nyvektor

	if (dist == 0)//nem osztunk 0-val
		return;

	v = (Q - (Origo * cosh(dist))) / sinh(dist);//ir?nyvektor a hiperbolikus s?kon

	//2 pont amire t?kr?z?nk ?gy hogy dist(m1, m2)=dist(Origo, Q)/2<--ez?rt van dist/4 ?s 3*dist/4 mivel igy 2/4 lesz a t?vols?guk
	vec3 m1 = (Origo * cosh(dist / 4)) + (v * sinh(dist / 4));//m1 az OriogoQ vektoron 
	vec3 m2 = (Origo * cosh(3*dist / 4)) + (v * sinh(3*dist / 4));//m2 az OriogoQ vektoron

	
	//az ?sszes pontunkat t?kr?zz?k az m1-re azt?n az m2-re
	for (int i = 0; i < 50; i++) {
		vec3 t = graph->vertices3D[i];//csak az?rt hogy kevesebbet kelljen ?rni a k?s?biekben

		float dist1 = acosh(-lorentz(m1, t));//m1 t t?vols?g
		if (dist1 != dist1)//Ha dist2 nan
			return;
		
		vec3 v1 = (m1 - (t * cosh(dist1))) / sinh(dist1);//ir?nyvektor t pontban
		vec3 t1 = (t * cosh(2 * dist1)) + (v1 * sinh(dist1 * 2));//t t?kr?zve m1-re

		float dist2 = acosh(-lorentz(m2, t1));//t1 m2 t?vols?g
		if (dist2 != dist2)//ha dist2 nan
			return;

		vec3 v2 = (m2 - (t1 * cosh(dist2))) / sinh(dist2);//ir?nyvektor t1 pontban
		vec3 t2 = (t1 * cosh(2 * dist2)) + (v2 * sinh(dist2 * 2));//t1 t?kr?zve m2-re

		graph->vertices3D[i] = t2;
		graph->vertices[i] = ConvertToVec2(graph->vertices3D[i]);
	}

	graph->InitNeighbors();
}

void ShiftOneNode(int i, vec2 MousePos) {
	vec3 Origo(0, 0, 1);
	vec3 Q = ConvertToVec3(MousePos);//A vektor amivel tolunk
	float dist = acosh(-lorentz(Origo, Q));//Origo ?s a Q t?vols?ga a hiperbolikus s?kon

	vec3 v;//ir?nyvektor

	if (dist == 0)//nem osztunk 0-val
		return;

	v = (Q - (Origo * cosh(dist))) / sinh(dist);//ir?nyvektor a hiperbolikus s?kon

	//2 pont amire t?kr?z?nk ?gy hogy dist(m1, m2)=dist(Origo, Q)/2<--ez?rt van dist/4 ?s 3*dist/4 mivel igy 2/4 lesz a t?vols?guk
	vec3 m1 = (Origo * cosh(dist / 4)) + (v * sinh(dist / 4));//m1 az OriogoQ vektoron 
	vec3 m2 = (Origo * cosh(3 * dist / 4)) + (v * sinh(3 * dist / 4));//m2 az OriogoQ vektoron

	vec3 t = graph->vertices3D[i];//csak az?rt hogy kevesebbet kelljen ?rni a k?s?biekben

	float dist1 = acosh(-lorentz(m1, t));//m1 t t?vols?g
	if (dist1 != dist1)//ha dist1 nan
		return;

	vec3 v1 = (m1 - (t * cosh(dist1))) / sinh(dist1);//ir?nyvektor t pontban
	vec3 t1 = (t * cosh(2 * dist1)) + (v1 * sinh(dist1 * 2));//t t?kr?zve m1-re

	float dist2 = acosh(-lorentz(m2, t1));//t1 m2 t?vols?g
	if (dist2 != dist2)//ha dist2 nan
		return;

	vec3 v2 = (m2 - (t1 * cosh(dist2))) / sinh(dist2);//ir?nyvektor t1 pontban
	vec3 t2 = (t1 * cosh(2 * dist2)) + (v2 * sinh(dist2 * 2));//t1 t?kr?zve m2-re

	graph->vertices3D[i] = t2;
	graph->vertices[i] = ConvertToVec2(graph->vertices3D[i]);
}

vec2 Direction(vec2 a, vec2 b) {
	return (a-b);
}

float Distance(vec3 a, vec3 b) {
	float dist = acosh(-lorentz(a, b));

	return dist;
}

vec2 ForceBetweenNeighbors(vec2 a, vec2 b, float param, float dist) {//ha szomsz?dok ez a kett? pont k?z?tt az er?
	vec2 temp = ((a - b) * (log(param)));// = log(d/d*)<-?gy j?ttem, r?, hogy ez a fv hasonl?t a legjobban a h?zi vide?ban l?v? f?gg?nyre
	return temp;
}

vec2 ForceBetweenVerts(vec2 a, vec2 b, float param, float dist) {//ha nem szomsz?dok ez a k?t pont k?z?tt az er?
	vec2 temp = ((a - b) * ((-1/(param*param))));// -1/(d / d*)^2 < -?gy j?ttem, r?, hogy ez a fv hasonl?t a legjobban a h?zi vide?ban l?v? f?gg?nyre
	return temp;
}


int iterations = 0;

void Sorting() {

	float idealDistance = 0.4f;

	for (int i = 0; i <50; i++) {

		vec2 force;//ered? er? egy pontra

		for (int j = 0; j < 50; j++) {
			if (i == j)
				continue;

			float dist = Distance(graph->vertices3D[i], graph->vertices3D[j]);//hiperbolikus s?kon a 2 vektor t?vols?ga

			float param = dist / idealDistance;

			if (neighborAlreadyExists(graph->indexes, i, j, 122)) {
				force = force + ForceBetweenNeighbors(graph->vertices[j], graph->vertices[i], param, dist);
			}
			else {
				force = force + ForceBetweenVerts(graph->vertices[j], graph->vertices[i], param, dist);
			}
		}

		force = force + (-graph->vertices[i] * 3.0f);//glob?lis er?t?r
		force = force * 0.01f;

		ShiftOneNode(i, force);//eltolja a pontot
	}

	graph->InitNeighbors();
	glutPostRedisplay();
}
bool sort = false;
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		graph->GetOptimalSet();
		sort = true;
	}
	glutPostRedisplay();
	
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

	glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long curr_time = glutGet(GLUT_ELAPSED_TIME);

	if (sort) {
		iterations++;

		Sorting();

		if (iterations == 1500) {
			sort = false;
			iterations = 0;
		}
	}
}