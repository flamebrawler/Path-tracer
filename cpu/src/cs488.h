// =======================================
// CS488/688 base code
// (written by Toshiya Hachisuka)
// =======================================
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX


// OpenGL
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>


// image loader and writer
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


// linear algebra 
#include "linalg.h"
using namespace linalg::aliases;


// animated GIF writer
#include "gif.h"

// misc
#include <iostream>
#include <vector>
#include <cfloat>

#include <queue>

#include <chrono>
// main window
static GLFWwindow* globalGLFWindow;


// window size and resolution
// (do not make it too large - will be slow!)
constexpr int globalWidth = 512;
constexpr int globalHeight = 384;


// degree and radian
constexpr float PI = 3.14159265358979f;
constexpr float DegToRad = PI / 180.0f;
constexpr float RadToDeg = 180.0f / PI;


// for ray tracing
constexpr float Epsilon = 5e-5f;


// amount the camera moves with a mouse and a keyboard
constexpr float ANGFACT = 0.2f;
constexpr float SCLFACT = 0.1f;


// fixed camera parameters
constexpr float globalAspectRatio = float(globalWidth / float(globalHeight));
constexpr float globalFOV = 45.0f; // vertical field of view
constexpr float globalDepthMin = Epsilon; // for rasterization
constexpr float globalDepthMax = 100.0f; // for rasterization
constexpr float globalFilmSize = 0.032f; //for ray tracing
const float globalDistanceToFilm = globalFilmSize / (2.0f * tan(globalFOV * DegToRad * 0.5f)); // for ray tracing


// particle system related
bool globalEnableParticles = false;
constexpr float deltaT = 0.002f;
constexpr float3 globalGravity = float3(0.0f, -9.8f, 0.0f);
constexpr int globalNumParticles = 100;


// dynamic camera parameters
float3 globalEye = float3(0.0f, 0.0f, 1.5f);
float3 globalLookat = float3(0.0f, 0.0f, 0.0f);
float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));
float3 globalViewDir; // should always be normalize(globalLookat - globalEye)
float3 globalRight; // should always be normalize(cross(globalViewDir, globalUp));
bool globalShowRaytraceProgress = true; // for ray tracing


// mouse event
static bool mouseLeftPressed;
static double m_mouseX = 0.0;
static double m_mouseY = 0.0;


// rendering algorithm
enum enumRenderType {
	RENDER_RASTERIZE,
	RENDER_RAYTRACE,
	RENDER_IMAGE,
};
enumRenderType globalRenderType = RENDER_IMAGE;
int globalFrameCount = 0;
static bool globalRecording = false;
static GifWriter globalGIFfile;
constexpr int globalGIFdelay = 1;


// OpenGL related data (do not modify it if it is working)
static GLuint GLFrameBufferTexture;
static GLuint FSDraw;
static const std::string FSDrawSource = R"(
    #version 120

    uniform sampler2D input_tex;
    uniform vec4 BufInfo;

    void main()
    {
        gl_FragColor = texture2D(input_tex, gl_FragCoord.st * BufInfo.zw);
    }
)";
static const char* PFSDrawSource = FSDrawSource.c_str();

void print(char* n,const float3 v[3]){
	for(int i = 0;i<3;i++)
		std::cout<<n<<i<<" "<<v[i].x<<" "<<v[i].y<<" "<<v[i].z<<std::endl;
}
void print(char* n,float3 v){
	std::cout<<n<<" "<<v.x<<" "<<v.y<<" "<<v.z<<std::endl;
}

// fast random number generator based pcg32_fast
#include <stdint.h>
namespace PCG32 {
	static uint64_t mcg_state = 0xcafef00dd15ea5e5u;	// must be odd
	static uint64_t const multiplier = 6364136223846793005u;
	uint32_t pcg32_fast(void) {
		uint64_t x = mcg_state;
		const unsigned count = (unsigned)(x >> 61);
		mcg_state = x * multiplier;
		x ^= x >> 22;
		return (uint32_t)(x >> (22 + count));
	}
	float rand() {
		return float(double(pcg32_fast()) / 4294967296.0);
	}
}



// image with a depth buffer
// (depth buffer is not always needed, but hey, we have a few GB of memory, so it won't be an issue...)
class Image {
public:
	std::vector<float3> pixels;
	std::vector<float> depths;
	int width = 0, height = 0;

	static float toneMapping(const float r) {
		// you may want to implement better tone mapping
		return std::max(std::min(1.0f, r), 0.0f);
	}

	static float gammaCorrection(const float r, const float gamma = 1.0f) {
		// assumes r is within 0 to 1
		// gamma is typically 2.2, but the default is 1.0 to make it linear
		return pow(r, 1.0f / gamma);
	}

	void resize(const int newWdith, const int newHeight) {
		this->pixels.resize(newWdith * newHeight);
		this->depths.resize(newWdith * newHeight);
		this->width = newWdith;
		this->height = newHeight;
	}

	void clear() {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				this->pixel(i, j) = float3(0.0f);
				this->depth(i, j) = FLT_MAX;
			}
		}
	}

	Image(int _width = 0, int _height = 0) {
		this->resize(_width, _height);
		this->clear();
	}

	bool valid(const int i, const int j) const {
		return (i >= 0) && (i < this->width) && (j >= 0) && (j < this->height);
	}

	float& depth(const int i, const int j) {
		return this->depths[i + j * width];
	}

	float3& pixel(const int i, const int j) {
		// optionally can check with "valid", but it will be slow
		return this->pixels[i + j * width];
	}

	void load(const char* fileName) {
		int comp, w, h;
		float* buf = stbi_loadf(fileName, &w, &h, &comp, 3);
		if (!buf) {
			std::cerr << "Unable to load: " << fileName << std::endl;
			return;
		}

		this->resize(w, h);
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				this->pixels[i + j * width] = float3(buf[k], buf[k + 1], buf[k + 2]);
				k += 3;
			}
		}
		delete[] buf;
		printf("Loaded \"%s\".\n", fileName);
	}
	void save(const char* fileName) {
		unsigned char* buf = new unsigned char[width * height * 3];
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).x)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).y)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).z)));
			}
		}
		stbi_write_png(fileName, width, height, 3, buf, width * 3);
		delete[] buf;
		printf("Saved \"%s\".\n", fileName);
	}
};

// main image buffer to be displayed
Image FrameBuffer(globalWidth, globalHeight);

// you may want to use the following later for progressive ray tracing
Image AccumulationBuffer(globalWidth, globalHeight);
unsigned int sampleCount = 0;



// keyboard events (you do not need to modify it unless you want to)
void keyFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
			case GLFW_KEY_R: {
				if (globalRenderType == RENDER_RAYTRACE) {
					printf("(Switched to rasterization)\n");
					glfwSetWindowTitle(window, "Rasterization mode");
					globalRenderType = RENDER_RASTERIZE;
				} else if (globalRenderType == RENDER_RASTERIZE) {
					printf("(Switched to ray tracing)\n");
					AccumulationBuffer.clear();
					sampleCount = 0;
					glfwSetWindowTitle(window, "Ray tracing mode");
					globalRenderType = RENDER_RAYTRACE;
				}
			break;}

			case GLFW_KEY_ESCAPE: {
				glfwSetWindowShouldClose(window, GL_TRUE);
			break;}

			case GLFW_KEY_I: {
				char fileName[1024];
				sprintf(fileName, "output%d.png", int(1000.0 * PCG32::rand()));
				FrameBuffer.save(fileName);
			break;}

			case GLFW_KEY_F: {
				if (!globalRecording) {
					char fileName[1024];
					sprintf(fileName, "output%d.gif", int(1000.0 * PCG32::rand()));
					printf("Saving \"%s\"...\n", fileName);
					GifBegin(&globalGIFfile, fileName, globalWidth, globalHeight, globalGIFdelay);
					globalRecording = true;
					printf("(Recording started)\n");
				} else {
					GifEnd(&globalGIFfile);
					globalRecording = false;
					printf("(Recording done)\n");
				}
			break;}

			case GLFW_KEY_W: {
				globalEye += SCLFACT * globalViewDir;
				globalLookat += SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_S: {
				globalEye -= SCLFACT * globalViewDir;
				globalLookat -= SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_Q: {
				globalEye += SCLFACT * globalUp;
				globalLookat += SCLFACT * globalUp;
			break;}

			case GLFW_KEY_Z: {
				globalEye -= SCLFACT * globalUp;
				globalLookat -= SCLFACT * globalUp;
			break;}

			case GLFW_KEY_A: {
				globalEye -= SCLFACT * globalRight;
				globalLookat -= SCLFACT * globalRight;
			break;}

			case GLFW_KEY_D: {
				globalEye += SCLFACT * globalRight;
				globalLookat += SCLFACT * globalRight;
			break;}

			default: break;
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void mouseButtonFunc(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouseLeftPressed = true;
		} else if (action == GLFW_RELEASE) {
			mouseLeftPressed = false;
			if (globalRenderType == RENDER_RAYTRACE) {
				AccumulationBuffer.clear();
				sampleCount = 0;
			}
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void cursorPosFunc(GLFWwindow* window, double mouse_x, double mouse_y) {
	if (mouseLeftPressed) {
		const float xfact = -ANGFACT * float(mouse_y - m_mouseY);
		const float yfact = -ANGFACT * float(mouse_x - m_mouseX);
		float3 v = globalViewDir;

		// local function in C++...
		struct {
			float3 operator()(float theta, const float3& v, const float3& w) {
				const float c = cosf(theta);
				const float s = sinf(theta);

				const float3 v0 = dot(v, w) * w;
				const float3 v1 = v - v0;
				const float3 v2 = cross(w, v1);

				return v0 + c * v1 + s * v2;
			}
		} rotateVector;

		v = rotateVector(xfact * DegToRad, v, globalRight);
		v = rotateVector(yfact * DegToRad, v, globalUp);
		globalViewDir = v;
		globalLookat = globalEye + globalViewDir;
		globalRight = cross(globalViewDir, globalUp);

		m_mouseX = mouse_x;
		m_mouseY = mouse_y;

		if (globalRenderType == RENDER_RAYTRACE) {
			AccumulationBuffer.clear();
			sampleCount = 0;
		}
	} else {
		m_mouseX = mouse_x;
		m_mouseY = mouse_y;
	}
}




class PointLightSource {
public:
	float3 position, wattage;
};



class Ray {
public:
	float3 o, d;
	Ray() : o(), d(float3(0.0f, 0.0f, 1.0f)) {}
	Ray(const float3& o, const float3& d) : o(o), d(d) {}
};



// uber material
// "type" will tell the actual type
// ====== implement it in A2, if you want ======
enum enumMaterialType {
	MAT_LAMBERTIAN,
	MAT_METAL,
	MAT_GLASS,
	MAT_LIGHT
};
class Material {
public:
	std::string name;

	enumMaterialType type = MAT_LAMBERTIAN;
	float eta = 1.0f;
	float glossiness = 1.0f;

	float3 Ka = float3(0.0f);
	float3 Kd = float3(0.9f);
	float3 Ks = float3(0.0f);
	float Ns = 0.0;

	// support 8-bit texture
	bool isTextured = false;
	unsigned char* texture = nullptr;
	int textureWidth = 0;
	int textureHeight = 0;

	Material() {};
	virtual ~Material() {};

	void setReflectance(const float3& c) {
		if (type == MAT_LAMBERTIAN) {
			Kd = c;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
	}

	float3 fetchTexture(const float2& tex) const {
		// repeating
		int x = int(tex.x * textureWidth) % textureWidth;
		int y = int(tex.y * textureHeight) % textureHeight;
		if (x < 0) x += textureWidth;
		if (y < 0) y += textureHeight;

		int pix = (x + y * textureWidth) * 3;
		const unsigned char r = texture[pix + 0];
		const unsigned char g = texture[pix + 1];
		const unsigned char b = texture[pix + 2];
		return float3(r, g, b) / 255.0f;
	}

	float3 BRDF(const float3& wi, const float3& wo, const float3& n) const {
		float3 brdfValue = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// BRDF
			brdfValue = Kd / PI;
		} else if (type == MAT_METAL) {
			brdfValue = Ks;
		} else if (type == MAT_GLASS) {
			// empty
		}
		return brdfValue;
	};

	float PDF(const float3& wGiven, const float3& wSample) const {
		// probability density function for a given direction and a given sample
		// it has to be consistent with the sampler
		float pdfValue = 0.0f;
		if (type == MAT_LAMBERTIAN) {
			pdfValue = 1/(PI);
		} else if (type == MAT_METAL) {
			pdfValue = 1;
		} else if (type == MAT_GLASS) {
			pdfValue = 1;
		}
		return pdfValue;
	}

	float3 sampler(const float3& wGiven, const float3& norm, float& pdfValue) const {
		// sample a vector and record its probability density as pdfValue
		float3 smp = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			float r1=2*PI*PCG32::rand(), r2=PCG32::rand(), r2s=sqrt(r2);
    		float3 w=norm;
			float3 u= normalize(cross(abs(w.x)>.1?float3(0,1,0):float3(1,0,0),w));
			float3 v=cross(w,u);
    		smp = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1-r2));
		} else if (type == MAT_METAL) {
			smp = normalize(-wGiven-2*dot(-wGiven,norm)*norm);
		} else if (type == MAT_GLASS) {
			// empty
		}

		pdfValue = PDF(wGiven, smp);
		return smp;
	}
};





class HitInfo {
public:
	float t; // distance
	float3 P; // location
	float3 N; // shading normal vector
	float2 T; // texture coordinate
	const Material* material; // const pointer to the material of the intersected object
};



// axis-aligned bounding box
class AABB {
private:
	float3 minp, maxp, size;

public:
	float3 get_minp() { return minp; };
	float3 get_maxp() { return maxp; };
	float3 get_size() { return size; };


	AABB() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	void reset() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	int getLargestAxis() const {
		if ((size.x > size.y) && (size.x > size.z)) {
			return 0;
		} else if (size.y > size.z) {
			return 1;
		} else {
			return 2;
		}
	}

	void fit(const float3& v) {
		if (minp.x > v.x) minp.x = v.x;
		if (minp.y > v.y) minp.y = v.y;
		if (minp.z > v.z) minp.z = v.z;

		if (maxp.x < v.x) maxp.x = v.x;
		if (maxp.y < v.y) maxp.y = v.y;
		if (maxp.z < v.z) maxp.z = v.z;

		size = maxp - minp;
	}

	float area() const {
		return (2.0f * (size.x * size.y + size.y * size.z + size.z * size.x));
	}


	bool intersect(HitInfo& minHit, const Ray& ray) const {
		// set minHit.t as the distance to the intersection point
		// return true/false if the ray hits or not
		float tx1 = (minp.x - ray.o.x) / ray.d.x;
		float ty1 = (minp.y - ray.o.y) / ray.d.y;
		float tz1 = (minp.z - ray.o.z) / ray.d.z;

		float tx2 = (maxp.x - ray.o.x) / ray.d.x;
		float ty2 = (maxp.y - ray.o.y) / ray.d.y;
		float tz2 = (maxp.z - ray.o.z) / ray.d.z;

		if (tx1 > tx2) {
			const float temp = tx1;
			tx1 = tx2;
			tx2 = temp;
		}

		if (ty1 > ty2) {
			const float temp = ty1;
			ty1 = ty2;
			ty2 = temp;
		}

		if (tz1 > tz2) {
			const float temp = tz1;
			tz1 = tz2;
			tz2 = temp;
		}

		float t1 = tx1; if (t1 < ty1) t1 = ty1; if (t1 < tz1) t1 = tz1;
		float t2 = tx2; if (t2 > ty2) t2 = ty2; if (t2 > tz2) t2 = tz2;

		if (t1 > t2) return false;
		if ((t1 < 0.0) && (t2 < 0.0)) return false;

		minHit.t = t1;
		return true;
	}
};




// triangle
struct Triangle {
	float3 positions[3];
	float3 normals[3];
	float2 texcoords[3];
	int idMaterial = 0;
	AABB bbox;
	float3 center;
};



// triangle mesh
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level = 0,const float eta = 1);
static float3 shade_pathtrace(const HitInfo& hit, const float3& viewDir, const int level=0,const float eta=1);
class TriangleMesh {
public:
	std::vector<Triangle> triangles;
	std::vector<Material> materials;
	AABB bbox;

	void transform(const float4x4& m) {
		// ====== implement it if you want =====
		// matrix transformation of an object	
		// m is a matrix that transforms an object
		// implement proper transformation for positions and normals
		// (hint: you will need to have float4 versions of p and n)
		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			for (int k = 0; k <= 2; k++) {
				const float3 &p = this->triangles[i].positions[k];
				const float3 &n = this->triangles[i].normals[k];

				const float4 &p_4 = float4(p,1);
				const float4 &n_4 = float4(n,1);

				// not doing anything right now
			}
		}
	}

	float getArea(float2 a,float2 b,float2 c)const{
		return abs(cross(c-a,b-a))/2.0f;
	}
	
	float3 getBaryCoords(std::array<float3,3> &tri, float2 coord)const{
		float3 result = {};
		for (int corner = 0;corner<3;corner++){
			result[corner] = getArea(tri[(corner+1)%3].xy(),tri[(corner+2)%3].xy(),coord);
		}
		return result;
	}

	bool belowLine(float3 point, float3 norm, float i,float j)const{
		return dot(float2({i,j})-point.xy(),norm.xy())<0;
	}


	void rasterizeTriangle(const Triangle& tri, const float4x4& plm) const {
		// ====== implement it in A1 ======
		// rasterization of a triangle
		// "plm" should be a matrix that contains perspective projection and the camera matrix
		// you do not need to implement clipping
		// you may call the "shade" function to get the pixel value
		// (you may ignore viewDir for now)

		
		std::array<float4,3> arr;
		std::array<float3,3> points;
		//std::cout<<"player pos: "<<globalEye.x<<" "<<globalEye.y<<" "<<globalEye.z<<std::endl;
		//std::cout<<"player lookat: "<<globalViewDir.x<<" "<<globalViewDir.y<<" "<<globalViewDir.z<<std::endl;
		for (int i = 0;i<3;i++){
			float4 x = float4(tri.positions[i],1);
			float4 b = float4(tri.normals[i],0);
			arr[i] = mul(plm,x);
			points[i] = arr[i].xyz()/arr[i].w;
			//std::cout<<"transformed: "<<arr[i].x<<" "<<arr[i].y<<" "<<arr[i].z<<" "<<arr[i].w<<std::endl;
			//std::cout<<"transformed/w: "<<points[i].x<<" "<<points[i].y<<" "<<points[i].z<<std::endl;
			//std::cout<<"untransformed: "<<x.x<<" "<<x.y<<" "<<x.z<<" "<<x.w<<std::endl;
		}
		//std::cout<<std::endl;


		float2 min = {(float)globalWidth,(float)globalHeight};
		float2 max = {0,0};

		//get bounds of triangle so you don't have to compute everything for all objects
		
		for (int i = 0;i<3;i++){
			for (int j = 0;j<2;j++){
				if (points[i][j]<min[j])
					min[j] = points[i][j] - 3.0f;// for rounding errors when sampling
				if (points[i][j]>max[j])
					max[j] = points[i][j] + 3.0f;
			}
		}
		
		
		//std::cout<<min.x<<" "<<min.y<<std::endl;
		//std::cout<<max.x<<" "<<max.y<<std::endl;

		std::array<float3,3> norms;

		//get normals for every side for calulating boundaries
		for (int i = 0;i<3;i++){
			float3 p1 = points[(i)%3];
			float3 p2 = points[(i+1)%3];
			float3 p3 = points[(i+2)%3];
			float3 a = p2-p1;
			//std::cout<<"a "<<a.x<<" "<<a.y<<std::endl;
			float3 c = p3-p1;
			//std::cout<<"c "<<c.x<<" "<<c.y<<std::endl;
			norms[i] = c- a*dot(a,c)/length2(a);
			//std::cout<<"norms "<<norms[i].x<<" "<<norms[i].y<<std::endl;
		}

		//float4 &refPoint1 = arr[2];
		//float3 normal = cross((arr[2]-arr[0]).xyz(),(arr[1]-arr[0]).xyz());

		float total = getArea(points[0].xy(),points[1].xy(),points[2].xy());

		float3 w = {1/arr[0].w,1/arr[1].w,1/arr[2].w};
		float3 z = {arr[0].z,arr[1].z,arr[2].z};
		float3 d = w*z;
		
		for (int j = std::max(0.0f,min.y); j < std::min(globalHeight,(int)max.y); j++) {
			for (int i = std::max(0.0f,min.x); i < std::min(globalWidth,(int)max.x); i++) {
				bool isInTriangle = true;
				for (int side = 0;side<3;side++){
					if (belowLine(points[side],norms[side],i,j)){
						isInTriangle = false;
					}
				}
					
				if (isInTriangle){
					float3 psi = getBaryCoords(points,{(float)i,(float)j})/total;

					//float depth = refPoint1.z - dot((float2({(float)i,(float)j}) - refPoint1.xy()),normal.xy())/normal.z;
					float depth = dot(psi,d);
					if(depth<1 && depth < FrameBuffer.depth(i,j)){
						HitInfo hit = {};
						hit.material = &materials[tri.idMaterial];
						hit.N = mul(float3x3(tri.normals[0],tri.normals[1],tri.normals[2]),psi);
						hit.P = float3({(float)i,(float)j,depth});
						hit.t = depth;
						if (hit.material->isTextured){
							const float2x3 truecorners ={tri.texcoords[0],tri.texcoords[1],tri.texcoords[2]};

							float2 P = mul(truecorners,psi*w);
							float W = dot(psi,w);
							float2 texcoord = P/W;
							hit.T = texcoord;
						}

						FrameBuffer.pixel(i, j) = shade(hit,{});//materials[tri.idMaterial].Kd ;
						FrameBuffer.depth(i,j) = depth;
					}
				}

			}
		}
	}


	bool raytraceTriangle(HitInfo& result, const Ray& ray, const Triangle& tri, float tMin, float tMax) const {
		// ====== implement it in A2 ======
		// ray-triangle intersection
		// fill in "result" when there is an intersection
		// return true/false if there is an intersection or not
		float3 a = tri.positions[0]- tri.positions[1];
		float3 b = tri.positions[0]-tri.positions[2];
		float3 c = ray.d;
		float3 d = tri.positions[0]-ray.o;
		
		float D = determinant(float3x3(a,b,c));
		float A = determinant(float3x3(d,b,c));
		float B = determinant(float3x3(a,d,c));
		float C = determinant(float3x3(a,b,d));
		
		float beta = A/D;
		float gamma = B/D;
		float alpha = 1-beta-gamma;
		//std::cout<<alpha<<" "<<beta<<" "<<gamma<<std::endl;

		if (alpha>1 || alpha<0 || beta<0 || gamma < 0)//outside of triangle
			return false;

		float3 phi = {alpha,beta,gamma};
		result.t = C/D;
		result.material = &materials[tri.idMaterial];
		result.P = mul(float3x3(tri.positions[0],tri.positions[1],tri.positions[2]),phi);
		result.N = mul(float3x3(tri.normals[0],tri.normals[1],tri.normals[2]),phi);
		result.T = mul(float2x3(tri.texcoords[0],tri.texcoords[1],tri.texcoords[2]),phi);

		if (dot(result.N,ray.d)>0){// if hit from the back
			result.N = -result.N;
		}
		//print("P",result.P);
		//print("tri",tri.positions);
		return result.t>tMin && result.t<tMax && result.t>0.00001;
	}


	// some precalculation for bounding boxes (you do not need to change it)
	void preCalc() {
		bbox.reset();
		for (int i = 0, _n = (int)triangles.size(); i < _n; i++) {
			this->triangles[i].bbox.reset();
			this->triangles[i].bbox.fit(this->triangles[i].positions[0]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[1]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[2]);

			this->triangles[i].center = (this->triangles[i].positions[0] + this->triangles[i].positions[1] + this->triangles[i].positions[2]) * (1.0f / 3.0f);

			this->bbox.fit(this->triangles[i].positions[0]);
			this->bbox.fit(this->triangles[i].positions[1]);
			this->bbox.fit(this->triangles[i].positions[2]);
		}
	}


	// load .obj file (you do not need to modify it unless you want to change something)
	bool load(const char* filename, const float4x4& ctm = linalg::identity) {
		int nVertices = 0;
		float* vertices;
		float* normals;
		float* texcoords;
		int nIndices;
		int* indices;
		int* matid = nullptr;

		printf("Loading \"%s\"...\n", filename);
		ParseOBJ(filename, nVertices, &vertices, &normals, &texcoords, nIndices, &indices, &matid);
		if (nVertices == 0) return false;
		this->triangles.resize(nIndices / 3);

		if (matid != nullptr) {
			for (unsigned int i = 0; i < materials.size(); i++) {
				// convert .mlt data into BSDF definitions
				// you may change the followings in the final project if you want
				materials[i].type = MAT_LAMBERTIAN;
				if (materials[i].Ns == 100.0f) {
					materials[i].type = MAT_METAL;
				}
				if (materials[i].name.compare(0, 5, "glass", 0, 5) == 0) {
					materials[i].type = MAT_GLASS;
					materials[i].eta = 1.5f;
				}
				if (materials[i].name.compare(0, 5, "light", 0, 5) == 0) {
					materials[i].type = MAT_LIGHT;
				}
			}
		} else {
			// use default Lambertian
			this->materials.resize(1);
		}

		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			const int v0 = indices[i * 3 + 0];
			const int v1 = indices[i * 3 + 1];
			const int v2 = indices[i * 3 + 2];

			this->triangles[i].positions[0] = float3(vertices[v0 * 3 + 0], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
			this->triangles[i].positions[1] = float3(vertices[v1 * 3 + 0], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
			this->triangles[i].positions[2] = float3(vertices[v2 * 3 + 0], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);

			if (normals != nullptr) {
				this->triangles[i].normals[0] = float3(normals[v0 * 3 + 0], normals[v0 * 3 + 1], normals[v0 * 3 + 2]);
				this->triangles[i].normals[1] = float3(normals[v1 * 3 + 0], normals[v1 * 3 + 1], normals[v1 * 3 + 2]);
				this->triangles[i].normals[2] = float3(normals[v2 * 3 + 0], normals[v2 * 3 + 1], normals[v2 * 3 + 2]);
			} else {
				// no normal data, calculate the normal for a polygon
				const float3 e0 = this->triangles[i].positions[1] - this->triangles[i].positions[0];
				const float3 e1 = this->triangles[i].positions[2] - this->triangles[i].positions[0];
				const float3 n = normalize(cross(e0, e1));

				this->triangles[i].normals[0] = n;
				this->triangles[i].normals[1] = n;
				this->triangles[i].normals[2] = n;
			}

			// material id
			this->triangles[i].idMaterial = 0;
			if (matid != nullptr) {
				// read texture coordinates
				if ((texcoords != nullptr) && materials[matid[i]].isTextured) {
					this->triangles[i].texcoords[0] = float2(texcoords[v0 * 2 + 0], texcoords[v0 * 2 + 1]);
					this->triangles[i].texcoords[1] = float2(texcoords[v1 * 2 + 0], texcoords[v1 * 2 + 1]);
					this->triangles[i].texcoords[2] = float2(texcoords[v2 * 2 + 0], texcoords[v2 * 2 + 1]);
				} else {
					this->triangles[i].texcoords[0] = float2(0.0f);
					this->triangles[i].texcoords[1] = float2(0.0f);
					this->triangles[i].texcoords[2] = float2(0.0f);
				}
				this->triangles[i].idMaterial = matid[i];
			} else {
				this->triangles[i].texcoords[0] = float2(0.0f);
				this->triangles[i].texcoords[1] = float2(0.0f);
				this->triangles[i].texcoords[2] = float2(0.0f);
			}
		}
		printf("Loaded \"%s\" with %d triangles.\n", filename, int(triangles.size()));

		delete[] vertices;
		delete[] normals;
		delete[] texcoords;
		delete[] indices;
		delete[] matid;

		return true;
	}

	~TriangleMesh() {
		materials.clear();
		triangles.clear();
	}


	bool bruteforceIntersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) {
		// bruteforce ray tracing (for debugging)
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		for (int i = 0; i < triangles.size(); ++i) {
			if (raytraceTriangle(tempMinHit, ray, triangles[i], tMin, tMax)) {
				if (tempMinHit.t < result.t) {
					hit = true;
					result = tempMinHit;
				}
			}
		}

		return hit;
	}

	void createSingleTriangle() {
		triangles.resize(1);
		materials.resize(1);

		triangles[0].idMaterial = 0;

		triangles[0].positions[0] = float3(-0.5f, -0.5f, 0.0f);
		triangles[0].positions[1] = float3(0.5f, -0.5f, 0.0f);
		triangles[0].positions[2] = float3(0.0f, 0.5f, 0.0f);

		const float3 e0 = this->triangles[0].positions[1] - this->triangles[0].positions[0];
		const float3 e1 = this->triangles[0].positions[2] - this->triangles[0].positions[0];
		const float3 n = normalize(cross(e0, e1));

		triangles[0].normals[0] = n;
		triangles[0].normals[1] = n;
		triangles[0].normals[2] = n;

		triangles[0].texcoords[0] = float2(0.0f, 0.0f);
		triangles[0].texcoords[1] = float2(0.0f, 1.0f);
		triangles[0].texcoords[2] = float2(1.0f, 0.0f);
	}


private:
	// === you do not need to modify the followings in this class ===
	void loadTexture(const char* fname, const int i) {
		int comp;
		materials[i].texture = stbi_load(fname, &materials[i].textureWidth, &materials[i].textureHeight, &comp, 3);
		if (!materials[i].texture) {
			std::cerr << "Unable to load texture: " << fname << std::endl;
			return;
		}
	}

	std::string GetBaseDir(const std::string& filepath) {
		if (filepath.find_last_of("/\\") != std::string::npos) return filepath.substr(0, filepath.find_last_of("/\\"));
		return "";
	}
	std::string base_dir;

	void LoadMTL(const std::string fileName) {
		FILE* fp = fopen(fileName.c_str(), "r");

		Material mtl;
		mtl.texture = nullptr;
		char line[81];
		while (fgets(line, 80, fp) != nullptr) {
			float r, g, b, s;
			std::string lineStr;
			lineStr = line;
			int i = int(materials.size());

			if (lineStr.compare(0, 6, "newmtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				mtl.name = lineStr;
				mtl.isTextured = false;
			} else if (lineStr.compare(0, 2, "Ka", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ka = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Kd", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Kd = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ks", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ks = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ns", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f\n", &s);
				mtl.Ns = s;
				mtl.texture = nullptr;
				materials.push_back(mtl);
			} else if (lineStr.compare(0, 6, "map_Kd", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				materials[i - 1].isTextured = true;
				loadTexture((base_dir + lineStr).c_str(), i - 1);
			}
		}

		fclose(fp);
	}

	void ParseOBJ(const char* fileName, int& nVertices, float** vertices, float** normals, float** texcoords, int& nIndices, int** indices, int** materialids) {
		// local function in C++...
		struct {
			void operator()(char* word, int* vindex, int* tindex, int* nindex) {
				const char* null = " ";
				char* ptr;
				const char* tp;
				const char* np;

				// by default, the texture and normal pointers are set to the null string
				tp = null;
				np = null;

				// replace slashes with null characters and cause tp and np to point
				// to character immediately following the first or second slash
				for (ptr = word; *ptr != '\0'; ptr++) {
					if (*ptr == '/') {
						if (tp == null) {
							tp = ptr + 1;
						} else {
							np = ptr + 1;
						}

						*ptr = '\0';
					}
				}

				*vindex = atoi(word);
				*tindex = atoi(tp);
				*nindex = atoi(np);
			}
		} get_indices;

		base_dir = GetBaseDir(fileName);
		#ifdef _WIN32
			base_dir += "\\";
		#else
			base_dir += "/";
		#endif

		FILE* fp = fopen(fileName, "r");
		int nv = 0, nn = 0, nf = 0, nt = 0;
		char line[81];
		if (!fp) {
			printf("Cannot open \"%s\" for reading\n", fileName);
			return;
		}

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (lineStr.compare(0, 6, "mtllib", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				LoadMTL(base_dir + lineStr);
			}

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					nn++;
				} else if (line[1] == 't') {
					nt++;
				} else {
					nv++;
				}
			} else if (line[0] == 'f') {
				nf++;
			}
		}
		fseek(fp, 0, 0);

		float* n = new float[3 * (nn > nf ? nn : nf)];
		float* v = new float[3 * nv];
		float* t = new float[2 * nt];

		int* vInd = new int[3 * nf];
		int* nInd = new int[3 * nf];
		int* tInd = new int[3 * nf];
		int* mInd = new int[nf];

		int nvertices = 0;
		int nnormals = 0;
		int ntexcoords = 0;
		int nindices = 0;
		int ntriangles = 0;
		bool noNormals = false;
		bool noTexCoords = false;
		bool noMaterials = true;
		int cmaterial = 0;

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					float x, y, z;
					sscanf(&line[2], "%f %f %f\n", &x, &y, &z);
					float l = sqrt(x * x + y * y + z * z);
					x = x / l;
					y = y / l;
					z = z / l;
					n[nnormals] = x;
					nnormals++;
					n[nnormals] = y;
					nnormals++;
					n[nnormals] = z;
					nnormals++;
				} else if (line[1] == 't') {
					float u, v;
					sscanf(&line[2], "%f %f\n", &u, &v);
					t[ntexcoords] = u;
					ntexcoords++;
					t[ntexcoords] = v;
					ntexcoords++;
				} else {
					float x, y, z;
					sscanf(&line[1], "%f %f %f\n", &x, &y, &z);
					v[nvertices] = x;
					nvertices++;
					v[nvertices] = y;
					nvertices++;
					v[nvertices] = z;
					nvertices++;
				}
			}
			if (lineStr.compare(0, 6, "usemtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				if (materials.size() != 0) {
					for (unsigned int i = 0; i < materials.size(); i++) {
						if (lineStr.compare(materials[i].name) == 0) {
							cmaterial = i;
							noMaterials = false;
							break;
						}
					}
				}

			} else if (line[0] == 'f') {
				char s1[32], s2[32], s3[32];
				int vI, tI, nI;
				sscanf(&line[1], "%s %s %s\n", s1, s2, s3);

				mInd[ntriangles] = cmaterial;

				// indices for first vertex
				get_indices(s1, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for second vertex
				get_indices(s2, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for third vertex
				get_indices(s3, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				ntriangles++;
			}
		}

		*vertices = new float[ntriangles * 9];
		if (!noNormals) {
			*normals = new float[ntriangles * 9];
		} else {
			*normals = 0;
		}

		if (!noTexCoords) {
			*texcoords = new float[ntriangles * 6];
		} else {
			*texcoords = 0;
		}

		if (!noMaterials) {
			*materialids = new int[ntriangles];
		} else {
			*materialids = 0;
		}

		*indices = new int[ntriangles * 3];
		nVertices = ntriangles * 3;
		nIndices = ntriangles * 3;

		for (int i = 0; i < ntriangles; i++) {
			if (!noMaterials) {
				(*materialids)[i] = mInd[i];
			}

			(*indices)[3 * i] = 3 * i;
			(*indices)[3 * i + 1] = 3 * i + 1;
			(*indices)[3 * i + 2] = 3 * i + 2;

			(*vertices)[9 * i] = v[3 * vInd[3 * i]];
			(*vertices)[9 * i + 1] = v[3 * vInd[3 * i] + 1];
			(*vertices)[9 * i + 2] = v[3 * vInd[3 * i] + 2];

			(*vertices)[9 * i + 3] = v[3 * vInd[3 * i + 1]];
			(*vertices)[9 * i + 4] = v[3 * vInd[3 * i + 1] + 1];
			(*vertices)[9 * i + 5] = v[3 * vInd[3 * i + 1] + 2];

			(*vertices)[9 * i + 6] = v[3 * vInd[3 * i + 2]];
			(*vertices)[9 * i + 7] = v[3 * vInd[3 * i + 2] + 1];
			(*vertices)[9 * i + 8] = v[3 * vInd[3 * i + 2] + 2];

			if (!noNormals) {
				(*normals)[9 * i] = n[3 * nInd[3 * i]];
				(*normals)[9 * i + 1] = n[3 * nInd[3 * i] + 1];
				(*normals)[9 * i + 2] = n[3 * nInd[3 * i] + 2];

				(*normals)[9 * i + 3] = n[3 * nInd[3 * i + 1]];
				(*normals)[9 * i + 4] = n[3 * nInd[3 * i + 1] + 1];
				(*normals)[9 * i + 5] = n[3 * nInd[3 * i + 1] + 2];

				(*normals)[9 * i + 6] = n[3 * nInd[3 * i + 2]];
				(*normals)[9 * i + 7] = n[3 * nInd[3 * i + 2] + 1];
				(*normals)[9 * i + 8] = n[3 * nInd[3 * i + 2] + 2];
			}

			if (!noTexCoords) {
				(*texcoords)[6 * i] = t[2 * tInd[3 * i]];
				(*texcoords)[6 * i + 1] = t[2 * tInd[3 * i] + 1];

				(*texcoords)[6 * i + 2] = t[2 * tInd[3 * i + 1]];
				(*texcoords)[6 * i + 3] = t[2 * tInd[3 * i + 1] + 1];

				(*texcoords)[6 * i + 4] = t[2 * tInd[3 * i + 2]];
				(*texcoords)[6 * i + 5] = t[2 * tInd[3 * i + 2] + 1];
			}

		}
		fclose(fp);

		delete[] n;
		delete[] v;
		delete[] t;
		delete[] nInd;
		delete[] vInd;
		delete[] tInd;
		delete[] mInd;
	}
};



// BVH node (for A2 extra)
class BVHNode {
public:
	bool isLeaf;
	int idLeft, idRight;
	int idParent;
	int idSibling;
	int triListNum;
	int* triList;
	AABB bbox;
};

// ====== implement it in A2 extra ======
// fill in the missing parts
class BVH {
public:
	const TriangleMesh* triangleMesh = nullptr;
	BVHNode* node = nullptr;

	const float costBBox = 1.0f;
	const float costTri = 1.0f;

	int leafNum = 0;
	int nodeNum = 0;

	BVH() {}
	virtual void build(const TriangleMesh* mesh);

	virtual bool intersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		// bvh
		if (this->node[0].bbox.intersect(tempMinHit, ray)) {
			hit = traverse(result, ray, 0, tMin, tMax);
		}
		if (result.t != FLT_MAX) hit = true;

		return hit;
	}
	bool traverse(HitInfo& result, const Ray& ray, int node_id, float tMin, float tMax) const;

protected:
	void sortAxis(int* obj_index, const char axis, const int li, const int ri) const;
	int splitBVH(int* obj_index, const int obj_num, const AABB& bbox);

};


// sort bounding boxes (in case you want to build SAH-BVH)
void BVH::sortAxis(int* obj_index, const char axis, const int li, const int ri) const {
	int i, j;
	float pivot;
	int temp;

	i = li;
	j = ri;

	pivot = triangleMesh->triangles[obj_index[(li + ri) / 2]].center[axis];

	while (true) {
		while (triangleMesh->triangles[obj_index[i]].center[axis] < pivot) {
			++i;
		}

		while (triangleMesh->triangles[obj_index[j]].center[axis] > pivot) {
			--j;
		}

		if (i >= j) break;

		temp = obj_index[i];
		obj_index[i] = obj_index[j];
		obj_index[j] = temp;

		++i;
		--j;
	}

	if (li < (i - 1)) sortAxis(obj_index, axis, li, i - 1);
	if ((j + 1) < ri) sortAxis(obj_index, axis, j + 1, ri);
}


static inline float SAHnosplitcost(const int obj_num){
	const float Cb = .1f, C0 = 1;
	return obj_num*C0;
}
static inline float SAHcost(const AABB& child, const AABB& parent,const int obj_num){
	const float Cb = .1f, C0 = 1;
	return Cb + child.area()/parent.area()*obj_num*C0;
}

#define SAHBVH // use this in once you have SAH-BVH
int BVH::splitBVH(int* obj_index, const int obj_num, const AABB& bbox) {
	// ====== exntend it in A2 extra ======
	AABB bestbboxL, bestbboxR;
	bool nosplit = false;
	int bestAxis, bestIndex;
#ifndef SAHBVH
	AABB bboxL, bboxR;
	int* sorted_obj_index  = new int[obj_num];

	// split along the largest axis
	bestAxis = bbox.getLargestAxis();

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

	// split in the middle
	bestIndex = obj_num / 2 - 1;

	bboxL.reset();
	for (int i = 0; i <= bestIndex; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxL.fit(tri.positions[0]);
		bboxL.fit(tri.positions[1]);
		bboxL.fit(tri.positions[2]);
	}

	bboxR.reset();
	for (int i = bestIndex + 1; i < obj_num; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxR.fit(tri.positions[0]);
		bboxR.fit(tri.positions[1]);
		bboxR.fit(tri.positions[2]);
	}

	bestbboxL = bboxL;
	bestbboxR = bboxR;

	if (obj_num<=4)
		nosplit = true;
#else
	// implelement SAH-BVH here
	AABB bboxL, bboxR;
	AABB* bboxLlist = new AABB[obj_num-1];
	AABB* bboxRlist = new AABB[obj_num-1];

	int* sorted_obj_index  = new int[obj_num];

	// split along the largest axis
	bestAxis = bbox.getLargestAxis();

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

	bboxL.reset();
	bboxR.reset();
	for (int i = 0; i < obj_num-1; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxL.fit(tri.positions[0]);
		bboxL.fit(tri.positions[1]);
		bboxL.fit(tri.positions[2]);
		bboxLlist[i] = bboxL;
		
		const Triangle& tri2 = triangleMesh->triangles[obj_index[obj_num-1-i]];
		bboxR.fit(tri2.positions[0]);
		bboxR.fit(tri2.positions[1]);
		bboxR.fit(tri2.positions[2]);
		bboxRlist[i] = bboxR;
	}

	
	float min_cost = FLT_MAX;

	for (int i = 0; i < obj_num-1; ++i) {
		AABB& bbox1 = bboxLlist[i];
		AABB& bbox2 = bboxRlist[obj_num-2-i];//bbox2 should contain the rest of the objects
		float cost1 = SAHcost(bbox1,bbox,i+1);
		float cost2 = SAHcost(bbox2,bbox,obj_num-i-1);
		float cost = cost1+cost2;
		if (cost<min_cost){
			bestbboxL = bboxLlist[i];
			bestbboxR = bboxRlist[obj_num-i-2];
			bestIndex= i;
			min_cost = cost;
		}
	}

	if (min_cost > SAHnosplitcost(obj_num) || obj_num <=1)
		nosplit = true;


#endif

	if (nosplit) {
		delete[] sorted_obj_index;

		this->nodeNum++;
		this->node[this->nodeNum - 1].bbox = bbox;
		this->node[this->nodeNum - 1].isLeaf = true;
		this->node[this->nodeNum - 1].triListNum = obj_num;
		#ifndef USING_GPU
		this->node[this->nodeNum - 1].triList = new int[obj_num];
		#else
		this->node[this->nodeNum - 1].triList = cudaMallocManaged(sizeof(int)*obj_num);
		#endif
		for (int i = 0; i < obj_num; i++) {
			this->node[this->nodeNum - 1].triList[i] = obj_index[i];
		}
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->leafNum++;

		return temp_id;
	} else {
		// split obj_index into two 
		int* obj_indexL = new int[bestIndex + 1];
		int* obj_indexR = new int[obj_num - (bestIndex + 1)];
		for (int i = 0; i <= bestIndex; ++i) {
			obj_indexL[i] = sorted_obj_index[i];
		}
		for (int i = bestIndex + 1; i < obj_num; ++i) {
			obj_indexR[i - (bestIndex + 1)] = sorted_obj_index[i];
		}
		delete[] sorted_obj_index;
		int obj_numL = bestIndex + 1;
		int obj_numR = obj_num - (bestIndex + 1);

		// recursive call to build a tree
		this->nodeNum++;
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->node[temp_id].bbox = bbox;
		this->node[temp_id].isLeaf = false;
		this->node[temp_id].idLeft = splitBVH(obj_indexL, obj_numL, bestbboxL);
		this->node[temp_id].idRight = splitBVH(obj_indexR, obj_numR, bestbboxR);

		delete[] obj_indexL;
		delete[] obj_indexR;

		return temp_id;
	}
}


// you may keep this part as-is
void BVH::build(const TriangleMesh* mesh) {
	triangleMesh = mesh;

	// construct the bounding volume hierarchy
	const int obj_num = (int)(triangleMesh->triangles.size());
	int* obj_index = new int[obj_num];
	for (int i = 0; i < obj_num; ++i) {
		obj_index[i] = i;
	}
	this->nodeNum = 0;
	#ifndef USING_GPU
	this->node = new BVHNode[obj_num * 2];
	#else
	this->node = cudaMallocManaged(sizeof(BVHNode)*obj_num * 2);
	#endif
	this->leafNum = 0;

	// calculate a scene bounding box
	AABB bbox;
	for (int i = 0; i < obj_num; i++) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];

		bbox.fit(tri.positions[0]);
		bbox.fit(tri.positions[1]);
		bbox.fit(tri.positions[2]);
	}

	// ---------- buliding BVH ----------
	printf("Building BVH...\n");
	splitBVH(obj_index, obj_num, bbox);
	printf("Done.\n");

	delete[] obj_index;
}


// you may keep this part as-is
bool BVH::traverse(HitInfo& minHit, const Ray& ray, int node_id, float tMin, float tMax) const {
	bool hit = false;
	HitInfo tempMinHit, tempMinHitL, tempMinHitR;
	bool hit1, hit2;

	if (this->node[node_id].isLeaf) {
		for (int i = 0; i < (this->node[node_id].triListNum); ++i) {
			if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[this->node[node_id].triList[i]], tMin, tMax)) {
				hit = true;
				if (tempMinHit.t < minHit.t) minHit = tempMinHit;
			}
		}
	} else {
		hit1 = this->node[this->node[node_id].idLeft].bbox.intersect(tempMinHitL, ray);
		hit2 = this->node[this->node[node_id].idRight].bbox.intersect(tempMinHitR, ray);

		hit1 = hit1 && (tempMinHitL.t < minHit.t);
		hit2 = hit2 && (tempMinHitR.t < minHit.t);

		if (hit1 && hit2) {
			if (tempMinHitL.t < tempMinHitR.t) {
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
			} else {
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
			}
		} else if (hit1) {
			hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
		} else if (hit2) {
			hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
		}
	}

	return hit;
}


struct CollNode{
	int idHit, idMiss;
};

class StacklessBVH : public BVH{
public:
	CollNode* refNode = nullptr;
	StacklessBVH() {}
	void build(TriangleMesh* mesh){
		BVH::build(mesh);
		refNode = new CollNode[nodeNum];
		
		std::function<void(int,int)> assign_hitmiss = [&](int missidx,int idx){
			if (node[idx].isLeaf)
				refNode[idx].idHit = missidx;
			else{
				refNode[idx].idHit = node[idx].idLeft;
				assign_hitmiss(node[idx].idRight,node[idx].idLeft);
				assign_hitmiss(missidx,node[idx].idRight);
			}
			refNode[idx].idMiss = missidx;
		};
		assign_hitmiss(0,0);
	}
	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX)const{

		HitInfo tempMinHit;
		int node_idx = 0;
		bool hit = false;
		minHit.t =FLT_MAX;

		do{	
			if (node[node_idx].bbox.intersect(tempMinHit, ray)) {
				if (node[node_idx].isLeaf) {
					for (int i = 0; i< node[node_idx].triListNum;i++)
						if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[node[node_idx].triList[i]], tMin, tMax)) {
							hit = true;
							if (tempMinHit.t < minHit.t) minHit = tempMinHit;
						}
				}
				node_idx = refNode[node_idx].idHit;
			} else {
				node_idx = refNode[node_idx].idMiss;
			}
		}while (node_idx!=0);
		return hit;
	}
	bool isHit(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX)const{

		HitInfo tempMinHit;
		int node_idx = 0;
		minHit.t =FLT_MAX;

		do{	
			if (node[node_idx].bbox.intersect(tempMinHit, ray)) {
				if (node[node_idx].isLeaf) {
					for (int i = 0; i< node[node_idx].triListNum;i++)
						if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[node[node_idx].triList[i]], tMin, tMax)) {
							if (tempMinHit.t < minHit.t) minHit = tempMinHit;
							return true;
						}
				}
				node_idx = refNode[node_idx].idHit;
			} else {
				node_idx = refNode[node_idx].idMiss;
			}
		}while (node_idx!=0);
		return false;
	}
};

class StacklessBVH2: public BVH{

public:
	
	StacklessBVH2() {}
	void build(TriangleMesh* mesh){
		BVH::build(mesh);

		node[0].idSibling = 0;
		node[0].idParent = 0;
		std::function<void(int)> assign_extra = [&](int idx){
			if (!node[idx].isLeaf){
				node[node[idx].idLeft].idSibling = node[idx].idRight;
				node[node[idx].idRight].idSibling = node[idx].idLeft;
				node[node[idx].idLeft].idParent = idx;
				node[node[idx].idRight].idParent = idx;

				assign_extra(node[idx].idLeft);
				assign_extra(node[idx].idRight);
			}
		};
		assign_extra(0);
	}

	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX)const{
		
		HitInfo tempMinHit;
		int node_idx = 0;
		int bittrail = 0;
		bool hit = false;
		minHit.t =FLT_MAX;

		HitInfo tempMinHitL, tempMinHitR;
		//assume the first one hits

		if(!node[node_idx].bbox.intersect(tempMinHit, ray)){
			return false;
		}
		
		while(true){
			//descend until you can't 
			while(!node[node_idx].isLeaf){
				bool hit1 = node[node[node_idx].idLeft].bbox.intersect(tempMinHitL, ray);
				bool hit2 = node[node[node_idx].idRight].bbox.intersect(tempMinHitR, ray);

				hit1 = hit1 && (tempMinHitL.t < minHit.t);
				hit2 = hit2 && (tempMinHitR.t < minHit.t);

				if (hit1 || hit2)
					bittrail<<=1;

				if (hit1 && hit2){
					bittrail^=1;
					if (tempMinHitL.t<tempMinHitR.t)
						node_idx = node[node_idx].idLeft;
					else
						node_idx = node[node_idx].idRight;
				} else if (hit1)
					node_idx = node[node_idx].idLeft;
				else if (hit2)
					node_idx = node[node_idx].idRight;
				else
					break;
			}
		
			if (node[node_idx].isLeaf) {
				for (int i = 0; i< node[node_idx].triListNum;i++)
					if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[node[node_idx].triList[i]], tMin, tMax)) {
						hit = true;
						if (tempMinHit.t < minHit.t) minHit = tempMinHit;
					}
			}
			while(true){
				if (bittrail==0)
					return hit;	

				//go up until you find a sibling to go to
				while((bittrail &1) ==0){
					bittrail>>=1;
					node_idx = node[node_idx].idParent;
				}

				if (bittrail & 0x1){//two hits from the parent
					node_idx = node[node_idx].idSibling;
					bittrail ^= 0x1;
					bool newhit = node[node_idx].bbox.intersect(tempMinHit, ray);
					if (!newhit)
						std::cout<<"this shouldnt happen"<<std::endl;
					if(tempMinHit.t<minHit.t)
						break;
				}
			}

		}
		return hit;
	}
};








// ====== implement it in A3 ======
// fill in the missing parts
class Particle {
public:
	float3 position = float3(0.0f);
	float3 velocity = float3(0.0f);
	float3 prevPosition = position;
	float3 netForce = float3(0.0f);
	float mass = 1;

	void reset() {
		position = float3(PCG32::rand(), PCG32::rand(), PCG32::rand()) - float(0.5f);
		velocity = float3(0.0f);//2.0f * float3((PCG32::rand() - 0.5f), 0.0f, (PCG32::rand() - 0.5f));
		prevPosition = position;
		position += velocity * deltaT;
	}

	int outsideBoundary()const{
		int ret =0;
		if(position.x<-.5 || position.x>.5) 
			ret|=1;
		if (position.y<-.5 || position.y>.5)
			ret|=2;
		if (position.z<-.5 || position.z>.5)
			ret|=4;
		return ret;
	}

	float reflectAcross(float p, float axis)const{
		return -(p-axis) + axis;
	}

	void bounceOffBoundaries(){
		int boundary = outsideBoundary();
		if(!boundary)
			return;	
		for (int i = 0;i<3;i++){
			if (boundary & (1<<i)){
				prevPosition[i] = reflectAcross(prevPosition[i],position[i]);
				float offset = position[i] - (position[i]<0?-.5:.5);
				prevPosition[i]-=offset;
				position[i]-=offset;
			}
		}
	}

	void mapToSphere(float3 c, float r){
		position = c+ r*(position - c)/length(position-c);
	}

	void step() {
		float3 temp = position;

		// === fill in this part in A3 ===
		// update the particle position and velocity here
		//netForce = globalGravity;
		float3 acceleration = netForce/mass;
		acceleration = globalGravity;
		position = position + (position - prevPosition) + deltaT*deltaT*acceleration;	
		prevPosition = temp;

		mapToSphere(float3(0.0f),.5);
		//bounceOffBoundaries();
	}

};


class ParticleSystem {
public:
	std::vector<Particle> particles;
	TriangleMesh particlesMesh;
	TriangleMesh sphere;
	const char* sphereMeshFilePath = 0;
	float sphereSize = 0;
	ParticleSystem() {};

	void updateMesh() {
		// you can optionally update the other mesh information (e.g., bounding box, BVH - which is tricky)
		if (sphereSize > 0) {
			const int n = int(sphere.triangles.size());
			for (int i = 0; i < globalNumParticles; i++) {
				for (int j = 0; j < n; j++) {
					particlesMesh.triangles[i * n + j].positions[0] = sphere.triangles[j].positions[0] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[1] = sphere.triangles[j].positions[1] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[2] = sphere.triangles[j].positions[2] + particles[i].position;
					particlesMesh.triangles[i * n + j].normals[0] = sphere.triangles[j].normals[0];
					particlesMesh.triangles[i * n + j].normals[1] = sphere.triangles[j].normals[1];
					particlesMesh.triangles[i * n + j].normals[2] = sphere.triangles[j].normals[2];
				}
			}
		} else {
			const float particleSize = 0.005f;
			for (int i = 0; i < globalNumParticles; i++) {
				// facing toward the camera
				particlesMesh.triangles[i].positions[0] = particles[i].position;
				particlesMesh.triangles[i].positions[1] = particles[i].position + particleSize * globalUp;
				particlesMesh.triangles[i].positions[2] = particles[i].position + particleSize * globalRight;
				particlesMesh.triangles[i].normals[0] = -globalViewDir;
				particlesMesh.triangles[i].normals[1] = -globalViewDir;
				particlesMesh.triangles[i].normals[2] = -globalViewDir;
			}
		}
	}

	void initialize() {
		particles.resize(globalNumParticles);
		particlesMesh.materials.resize(1);
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].reset();
		}
		//particles[0].mass = 10;

		if (sphereMeshFilePath) {
			if (sphere.load(sphereMeshFilePath)) {
				particlesMesh.triangles.resize(sphere.triangles.size() * globalNumParticles);
				sphere.preCalc();
				sphereSize = sphere.bbox.get_size().x * 0.5f;
			} else {
				particlesMesh.triangles.resize(globalNumParticles);
			}
		} else {
			particlesMesh.triangles.resize(globalNumParticles);
		}
		updateMesh();
	}

	void computeGravity(){
		for(int i =0;i<globalNumParticles;i++){
			particles[i].netForce = float3(0.0f);
			for(int j =1;j<globalNumParticles;j++){
				if (i==j)
					continue;
				float G = 2e-3;
				Particle& p1 = particles[i];
				Particle& p2 = particles[j];
				float3 force = G*p1.mass*p2.mass*(p1.position-p2.position)/((float)std::pow(length(p1.position-p2.position),3)+0.000001f);
				p1.netForce-= force;
				//p2.netForce+= force;
			}
		}
	}
	void computeCollisions(){
		if(sphereSize==0)
			return;
		bool hasCollisions = true;
		int reps = 0;
		do{
			hasCollisions = false;
			for (int i = 0;i<globalNumParticles;i++){
				for (int j = i+1;j<globalNumParticles;j++){
					Particle& p1 = particles[i];
					Particle& p2 = particles[j];
					float diff = length(p1.position-p2.position);
					if(diff<sphereSize*2){
						hasCollisions=true;
						float overlap = (sphereSize*2-diff);
						float3 offset = (p1.position-p2.position)/diff*overlap/2;
						float prevDist = length(p1.position-p1.prevPosition)+length(p2.position-p2.prevPosition);
						p1.position += offset;
						p1.prevPosition = p1.position - offset/length(offset)*prevDist/2;
						p2.position -= offset;
						p2.prevPosition = p2.position + offset/length(offset)*prevDist/2;
					}
				}
			}
			reps++;

		}while(hasCollisions && reps<1000);
	}
	void step() {
		// add some particle-particle interaction here
		// spherical particles can be implemented here
		//computeGravity();
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].step();
		}
		//computeCollisions();
		updateMesh();
	}
};
static ParticleSystem globalParticleSystem;








// scene definition
class Scene {
public:
	std::vector<TriangleMesh*> objects;
	std::vector<PointLightSource*> pointLightSources;
	std::vector<StacklessBVH2> bvhs;
	Image background;
	static const int total_angles_h = 0;
	static const int total_angles_v = 0;
	float3 directions[total_angles_h*total_angles_v];

	void computeDirections(){
		for ( int i = 0;i<total_angles_h;i++)
			for ( int j = 0;j<total_angles_v;j++)
				directions[i+j*total_angles_h] = normalize(float3(sin(i/(total_angles_h)*3.14159f),j/(total_angles_v-1)/2.0f+.5f,cos(i/(total_angles_h)*3.14159f)));
	}

	void addObject(TriangleMesh* pObj) {
		objects.push_back(pObj);
	}
	void addLight(PointLightSource* pObj) {
		pointLightSources.push_back(pObj);
	}

	void preCalc() {
		
		computeDirections();
		bvhs.resize(objects.size());
		for (int i = 0; i < objects.size(); i++) {
			objects[i]->preCalc();
			bvhs[i].build(objects[i]);
		}
		
	}

	float3 getBackground(float3 dir) const{
		float r = acos(dir.z)/6.283184f;
		float2 xy = normalize(dir.xy());
		float2 tex = {r*xy.x+.5f,-r*xy.y+.5f};

		int x = int(tex.x * background.width) % background.width;
		int y = int(tex.y * background.height) % background.height;
		if (x < 0) x += background.width;
		if (y < 0) y += background.height;

		int pix = (x + y * background.width);


		float3 pixel = background.pixels[pix];
		return pixel;
	}

	// faster intersect if you don't need to know where it hits
	bool isHit(const HitInfo& excl, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;

		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, ray, tMin, tMax)) {
				hit = true;
				break;
			}
		}
		return hit;
	}
	// ray-scene intersection
	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		minHit.t = FLT_MAX;

		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, ray, tMin, tMax)) {
				if (tempMinHit.t < minHit.t) {
					hit = true;
					minHit = tempMinHit;
				}
			}
		}
		return hit;
	}

	// camera -> screen matrix (given to you for A1)
	float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar) const {
		float4x4 m;
		const float f = 1.0f / (tan(fovy * DegToRad / 2.0f));
		m[0] = { f / aspect, 0.0f, 0.0f, 0.0f };
		m[1] = { 0.0f, f, 0.0f, 0.0f };
		m[2] = { 0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), -1.0f };
		m[3] = { 0.0f, 0.0f, (2.0f * zFar * zNear) / (zNear - zFar), 0.0f };

		return m;
	}
	float4x4 screenMapMatrix(const int screenWidth, const int screenHeight) const {
		// transformation to the camera coordinate
		const float w = (float)screenWidth;
		const float h = (float)screenHeight;

		float4x4 m;
		m[0] = { w, 0.0f, 0.0f, 0.0f };
		m[1] = { 0.0f, h, 0.0f, 0.0f };
		m[2] = { 0.0f, 0.0f, 1.0f, 0.0f };
		m[3] = { 0.0f, 0.0f, 0.0f, 1.0f };

		// translation according to the camera location
		const float4x4 t = float4x4{ 
			{1.0f, 0.0f, 0.0f, 0.0f},
			{0.0f, 1.0f, 0.0f, 0.0f},
			{0.0f, 0.0f, 1.0f, 0.0f},
			{ 1.0f, 1.0f, 0.0f, 1.0f} };

		m = mul(m, t);
		return m;
	}

	// model -> camera matrix (given to you for A1)
	float4x4 lookatMatrix(const float3& _eye, const float3& _center, const float3& _up) const {
		// transformation to the camera coordinate
		float4x4 m;
		const float3 f = normalize(_center - _eye);
		const float3 upp = normalize(_up);
		const float3 s = normalize(cross(f, upp));
		const float3 u = cross(s, f);

		m[0] = { s.x, s.y, s.z, 0.0f };
		m[1] = { u.x, u.y, u.z, 0.0f };
		m[2] = { -f.x, -f.y, -f.z, 0.0f };
		m[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
		m = transpose(m);

		// translation according to the camera location
		const float4x4 t = float4x4{ {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, { -_eye.x, -_eye.y, -_eye.z, 1.0f} };

		m = mul(m, t);
		return m;
	}

	// rasterizer
	void Rasterize() const {
		// ====== implement it in A1 ======
		// fill in plm by a proper matrix
		const float4x4 pm = perspectiveMatrix(globalFOV, globalAspectRatio, globalDepthMin, globalDepthMax);
		const float4x4 lm = lookatMatrix(globalEye, globalLookat, globalUp);
		const float4x4 plm = mul(pm, lm);
		
		const float4x4 sm = screenMapMatrix(globalWidth,globalHeight);
		const float4x4 fm = mul(sm,plm);
/*
		for (int row = 0;row<4;row++){
			for (auto el : pm.row(row))
				std::cout<<el<<" ";
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
		for (int row = 0;row<4;row++){
			for (auto el : lm.row(row))
				std::cout<<el<<" ";
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
		for (int row = 0;row<4;row++){
			for (auto el : plm.row(row))
				std::cout<<el<<" ";
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
		 */
			
		FrameBuffer.clear();
		FrameBuffer.pixels[0]= {1,1,1};
		for (int n = 0, n_n = (int)objects.size(); n < n_n; n++) {
			for (int k = 0, k_n = (int)objects[n]->triangles.size(); k < k_n; k++) {
				objects[n]->rasterizeTriangle(objects[n]->triangles[k], fm);
			}
		}
	}

	// eye ray generation (given to you for A2)
	Ray eyeRay(int x, int y) const {
		// compute the camera coordinate system 
		const float3 wDir = normalize(float3(-globalViewDir));
		const float3 uDir = normalize(cross(globalUp, wDir));
		const float3 vDir = cross(wDir, uDir);

		// compute the pixel location in the world coordinate system using the camera coordinate system
		// trace a ray through the center of each pixel
		const float imPlaneUPos = (x + 0.5f) / float(globalWidth) - 0.5f;
		const float imPlaneVPos = (y + 0.5f) / float(globalHeight) - 0.5f;

		const float3 pixelPos = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir + float(globalFilmSize * imPlaneVPos) * vDir - globalDistanceToFilm * wDir;

		return Ray(globalEye, normalize(pixelPos - globalEye));
	}

	// ray tracing (you probably don't need to change it in A2)
	void Raytrace() const {
		//FrameBuffer.clear();

		// loop over all pixels in the image
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = 0; i < globalWidth; ++i) {
				const Ray ray = eyeRay(i, j);
				HitInfo hitInfo;
				if (intersect(hitInfo, ray)) {
					FrameBuffer.pixel(i, j) = shade(hitInfo, -ray.d);
				} else {
					//FrameBuffer.pixel(i,j) = float3(0.0f);
					FrameBuffer.pixel(i, j) = getBackground(-ray.d);
				}
			}

			// show intermediate process
			if (globalShowRaytraceProgress) {
				constexpr int scanlineNum = 64;
				//printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
				if ((j % scanlineNum) == (scanlineNum - 1)) {
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
					glRecti(1, 1, -1, -1);
					glfwSwapBuffers(globalGLFWindow);
					//printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
					fflush(stdout);
				}
			}
		}
	}
	void Pathtrace() const {
		//FrameBuffer.clear();
		sampleCount++;
		// loop over all pixels in the image
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = 0; i < globalWidth; ++i) {
				const Ray ray = eyeRay(i, j);
				HitInfo hitInfo;
				if (intersect(hitInfo, ray)) {
					float3 result = shade_pathtrace(hitInfo, -ray.d);
					AccumulationBuffer.pixel(i, j) += result;
					FrameBuffer.pixel(i, j) = AccumulationBuffer.pixel(i,j)/sampleCount;
				} else {
					//FrameBuffer.pixel(i,j) = float3(0.0f);
					FrameBuffer.pixel(i, j) = getBackground(-ray.d);
				}
			}


			// show intermediate process
			if (globalShowRaytraceProgress) {
				constexpr int scanlineNum = 64;
				if ((j % scanlineNum) == (scanlineNum - 1)) {
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
					glRecti(1, 1, -1, -1);
					glfwSwapBuffers(globalGLFWindow);
					//printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
					fflush(stdout);
				}
			}
		}
	}

};
static Scene globalScene;


static float3 shade_pathtrace(const HitInfo& hit, const float3& viewDir, const int level,const float eta) {

	float3 currDir = viewDir;
	float3 multTerm = float3(1);
	float3 constTerm = float3(0.0f);
	int currlevel = level;
	HitInfo recHit = hit;
	float currEta = eta;

	while (true){
		if (recHit.material->type == MAT_LAMBERTIAN) {
			// you may want to add shadow ray tracing here in A2
			float3 L = float3(0.0f);
			float3 brdf, irradiance;

			float3 texture = float3(1);
			if (recHit.material->isTextured) {
				texture *= recHit.material->fetchTexture(recHit.T);
			}
			// loop over all of the point light sources
			for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
				float3 l = globalScene.pointLightSources[i]->position - recHit.P;


				// the inverse-squared falloff
				const float falloff = length2(l);
				HitInfo tempHit;
				//don't do anything if there is an object in the way
				// normalize the light direction
				l /= sqrtf(falloff);
				

				brdf = recHit.material->BRDF(l, currDir, recHit.N);
				// get the irradiance
				irradiance = float(std::max(0.0f, dot(recHit.N, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
				//check if light is on the side you can see and that there is nothing abstructing the light
				if (dot(recHit.N,l)>0 && globalScene.isHit(recHit,Ray(recHit.P,normalize(l)),0,sqrtf(falloff))){
					//irradiance= float3(0.5f)*irradiance;
					continue;
				}
				//return brdf * PI; //debug output

				L += irradiance * texture*brdf;
			}
			if (currlevel>5){
				if(PCG32::rand()<.25f)
					return constTerm + multTerm*L;
			}
			
			float pdf;
			float3 sample = recHit.material->sampler(currDir,recHit.N,pdf);
			float3 sample_brdf = recHit.material->BRDF(sample,currDir,recHit.N);
			HitInfo tempHit;
			bool hasHit = globalScene.intersect(tempHit,Ray(recHit.P,sample),0.001);
			if (hasHit){
				constTerm += L*multTerm;
				multTerm *= texture*sample_brdf/pdf;
				currDir = sample;
				currlevel +=1;
				recHit = tempHit;

				//shade_pathtrace(tempHit,sample,level+1,eta);
			}else{
				
				//*dot(sample,viewDir)/(length(sample)*length(viewDir))/pdf;

				return constTerm + multTerm*(L/*+texture*sample_brdf*/);
			}
			
			//texture*sample_brdf*globalScene.getBackground(sample);
			
			//return L;
		} else if (recHit.material->type == MAT_METAL) {
		
			float pdf;
			float3 sample = recHit.material->sampler(currDir,recHit.N,pdf);
			float3 sample_brdf = recHit.material->BRDF(sample,currDir,recHit.N);
			HitInfo tempHit;
			bool hasHit = globalScene.intersect(tempHit,Ray(recHit.P,sample),0.001);

			if (hasHit){
				multTerm*=sample_brdf;
				currDir = sample;
				currlevel+=1;
				recHit = tempHit;
				//return sample_brdf*shade_pathtrace(tempHit,sample,level+1,eta);//*dot(sample,viewDir)/(length(sample)*length(viewDir))/pdf;
			}else
				return constTerm + multTerm*globalScene.getBackground(sample);
			
		} else if (recHit.material->type == MAT_GLASS) {
				//if (level>5)//make it only go through 2 mirrors
				//	return float3(0,0,0);
				float eta2 = recHit.material->eta ==currEta?1:recHit.material->eta;
				float3 w = -currDir;
				float3 n = recHit.N; 
				float inside = 1-pow(currEta/eta2,2)*(1-pow(dot(w,n),2));
				HitInfo tempHit;


				if(inside>=0){
					float3 refracted = -normalize(currEta/eta2*(w - dot(w,n)*n)- (sqrtf(inside))*n);
					bool hasHit = globalScene.intersect(tempHit,Ray(recHit.P,-refracted),0.01);

					if (!hasHit)
						return constTerm + multTerm*globalScene.getBackground(-refracted);
						//return float3(0,0,0);
					currDir = refracted;
					currlevel+=1;
					recHit = tempHit;
					currEta = eta2;
					//return shade_pathtrace(tempHit,-refracted,level+1,eta2);
				}else{
					HitInfo tempHit;
					float3 reflected = normalize(w-2*dot(w,n)*n);
					bool hasHit = globalScene.intersect(tempHit,Ray(recHit.P,reflected));
					if (!hasHit)
						return constTerm + multTerm*globalScene.getBackground(reflected);
						//return globalScene.getBackground(-reflected);
					//return shade_pathtrace(tempHit,reflected,level+1,eta);
					currDir = reflected;
					currlevel+=1;
					recHit = tempHit;
				}

		} else {
			// something went wrong - make it apparent that it is an error
			return float3(0, 0.0f, 0);
		}
		
	}
}


// ====== implement it in A2 ======
// fill in the missing parts
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level,const float eta) {
	if (hit.material->type == MAT_LAMBERTIAN) {
		// you may want to add shadow ray tracing here in A2
		float3 L = float3(0.0f);
		float3 brdf, irradiance;

		// loop over all of the point light sources
		for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
			float3 l = globalScene.pointLightSources[i]->position - hit.P;


			// the inverse-squared falloff
			const float falloff = length2(l);
			HitInfo tempHit;
			//don't do anything if there is an object in the way
			// normalize the light direction
			l /= sqrtf(falloff);
			

			// get the irradiance
			irradiance = float(std::max(0.0f, dot(hit.N, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
			brdf = hit.material->BRDF(l, viewDir, hit.N);
			//check if light is on the side you can see and that there is nothing abstructing the light
			if (hit.material->isTextured) {
				brdf *= hit.material->fetchTexture(hit.T);
			}
			if (dot(hit.N,l)>0 && globalScene.isHit(hit,Ray(hit.P,normalize(l)),0,sqrtf(falloff))){
				irradiance= float3(0.1f)*irradiance;
				//continue;
			}
			//return brdf * PI; //debug output

			L += irradiance * brdf;
		}
		
		int xmax = globalScene.total_angles_h, ymax = globalScene.total_angles_v;
		for(int i = 0;i<xmax;i++){//go through all the angles
			for(int j = 0;j<ymax;j++){
				float3 l = globalScene.directions[i+j*xmax];
				if (dot(hit.N,l)>0 && globalScene.isHit(hit,Ray(hit.P,normalize(l)))){
					continue;
				}
				irradiance = float(std::max(0.0f, dot(hit.N, l)) / (4.0 * PI)) * 100.0f*globalScene.getBackground(l)/xmax/ymax;
				brdf = hit.material->BRDF(l, viewDir, hit.N);
				if (hit.material->isTextured) {
					brdf *= hit.material->fetchTexture(hit.T);
				}
				L += irradiance * brdf;
			}
		}
		
		return L;
	} else if (hit.material->type == MAT_METAL) {
	
		if (level>5)//make it only go through 2 mirrors`
			return float3(.1,.1,.1);
		HitInfo tempHit;
		float3 reflected = normalize(-viewDir-2*dot(-viewDir,hit.N)*hit.N);
		bool hasHit = globalScene.intersect(tempHit,Ray(hit.P,reflected),0.001);
		if (!hasHit)
			return globalScene.getBackground(-reflected);
		return hit.material->Ks*shade(tempHit,reflected,level+1,eta);
		
	} else if (hit.material->type == MAT_GLASS) {
			if (level>5)//make it only go through 2 mirrors
				return float3(0,0,0);
		 	float eta2 = hit.material->eta ==eta?1:hit.material->eta;
			float3 w = -viewDir;
			float3 n = hit.N; 
			float inside = 1-pow(eta/eta2,2)*(1-pow(dot(w,n),2));
			HitInfo tempHit;

			if(inside>=0){
				float3 refracted = normalize(eta/eta2*(w - dot(w,n)*n)- (sqrtf(inside))*n);
				bool hasHit = globalScene.intersect(tempHit,Ray(hit.P,refracted));
				if (!hasHit)
					return globalScene.getBackground(-refracted);
				 	//return float3(0,0,0);
				return 	shade(tempHit,-refracted,level+1,eta2);
			}else{
				//if (level>6)//make it only go through 2 mirrors
			//		return float3(.1,0,.1);
				HitInfo tempHit;
				float3 reflected = normalize(w-2*dot(w,n)*n);
				bool hasHit = globalScene.intersect(tempHit,Ray(hit.P,reflected));
				if (!hasHit)
					return globalScene.getBackground(-reflected);
				return shade(tempHit,reflected,level+1,eta);
			}

	} else {
		// something went wrong - make it apparent that it is an error
		return float3(100.0f, 0.0f, 100.0f);
	}
}







// OpenGL initialization (you will not use any OpenGL/Vulkan/DirectX... APIs to render 3D objects!)
// you probably do not need to modify this in A0 to A3.
class OpenGLInit {
public:
	OpenGLInit() {
		// initialize GLFW
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW." << std::endl;
			exit(-1);
		}

		// create a window
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		globalGLFWindow = glfwCreateWindow(globalWidth, globalHeight, "Welcome to CS488/688!", NULL, NULL);
		if (globalGLFWindow == NULL) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// make OpenGL context for the window
		glfwMakeContextCurrent(globalGLFWindow);

		// initialize GLEW
		glewExperimental = true;
		if (glewInit() != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// set callback functions for events
		glfwSetKeyCallback(globalGLFWindow, keyFunc);
		glfwSetMouseButtonCallback(globalGLFWindow, mouseButtonFunc);
		glfwSetCursorPosCallback(globalGLFWindow, cursorPosFunc);

		// create shader
		FSDraw = glCreateProgram();
		GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(s, 1, &PFSDrawSource, 0);
		glCompileShader(s);
		glAttachShader(FSDraw, s);
		glLinkProgram(FSDraw);

		// create texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &GLFrameBufferTexture);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, globalWidth, globalHeight, 0, GL_LUMINANCE, GL_FLOAT, 0);

		// initialize some OpenGL state (will not change)
		glDisable(GL_DEPTH_TEST);

		glUseProgram(FSDraw);
		glUniform1i(glGetUniformLocation(FSDraw, "input_tex"), 0);

		GLint dims[4];
		glGetIntegerv(GL_VIEWPORT, dims);
		const float BufInfo[4] = { float(dims[2]), float(dims[3]), 1.0f / float(dims[2]), 1.0f / float(dims[3]) };
		glUniform4fv(glGetUniformLocation(FSDraw, "BufInfo"), 1, BufInfo);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	virtual ~OpenGLInit() {
		glfwTerminate();
	}
};



// main window
// you probably do not need to modify this in A0 to A3.
class CS488Window {
public:
	// put this first to make sure that the glInit's constructor is called before the one for CS488Window
	OpenGLInit glInit;

	CS488Window() {}
	virtual ~CS488Window() {}

	void(*process)() = NULL;

	void start() const {
		if (globalEnableParticles) {
			globalScene.addObject(&globalParticleSystem.particlesMesh);
		}
		globalScene.preCalc();
		int samples = 0;
		long long total = 0;
		// main loop
		while (glfwWindowShouldClose(globalGLFWindow) == GL_FALSE) {
			auto start = std::chrono::high_resolution_clock::now();
			glfwPollEvents();
			globalViewDir = normalize(globalLookat - globalEye);
			globalRight = normalize(cross(globalViewDir, globalUp));

			if (globalEnableParticles) {
				globalParticleSystem.step();
			}
			//AccumulationBuffer.clear();
			if (globalRenderType == RENDER_RASTERIZE) {
				globalScene.Rasterize();
			} else if (globalRenderType == RENDER_RAYTRACE) {
				//globalScene.Pathtrace();
				globalScene.Raytrace();
			} else if (globalRenderType == RENDER_IMAGE) {
				if (process) process();
			}

			if (globalRecording) {
				unsigned char* buf = new unsigned char[FrameBuffer.width * FrameBuffer.height * 4];
				int k = 0;
				for (int j = FrameBuffer.height - 1; j >= 0; j--) {
					for (int i = 0; i < FrameBuffer.width; i++) {
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).x));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).y));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).z));
						buf[k++] = 255;
					}
				}
				GifWriteFrame(&globalGIFfile, buf, globalWidth, globalHeight, globalGIFdelay);
				delete[] buf;
			}

			// drawing the frame buffer via OpenGL (you don't need to touch this)
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0][0]);
			glRecti(1, 1, -1, -1);
			glfwSwapBuffers(globalGLFWindow);
			globalFrameCount++;
			PCG32::rand();
			auto end = std::chrono::high_resolution_clock::now();
			auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
			total+=frame_duration;
			printf("Frame duration %lld, average %lld, sample %d\n",frame_duration,(total/globalFrameCount),globalFrameCount);
		}
	}
};


