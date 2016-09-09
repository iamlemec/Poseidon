#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "Poseidon_kernel.h"

// Random number macros
#define RANDOMSEED(seed) ((seed) = ((seed) * 1103515245 + 12345))
#define RANDOMBITS(seed, bits) ((unsigned int)RANDOMSEED(seed) >> (32 - (bits)))

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

//Source image on the host side
uchar4 *h_Src = 0;

// Destination image on the GPU side
uchar4 *d_dst = NULL;
uchar4 *d_dst_old = NULL;
float4 *d_field = NULL;

//Original image width and height
int imageW = 1000, imageH = 800;

// Starting color multipliers and random seed
int colorSeed = 0;
uchar4 colors;

// User interface variables
bool leftClicked = false;
bool middleClicked = false;
bool rightClicked = false;

bool running = true;
bool frame = false;
bool pulse = false;
int px = 0;
int py = 0;
int fx = 0;
int fy = 0;
bool init = true;
bool regen = true;

// Timer ID
float speed = 10.0;

//float rfact = 0.4;
//float tfact = 0.19;
float rfact = 2.063911;
float tfact = 0.059313;

int fieldType = 1;

#define MAX_EPSILON 50

#define MAX(a,b) ((a > b) ? a : b)

#define BUFFER_DATA(i) ((char *)0 + i)

void renderImage()
{
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &num_bytes, cuda_pbo_resource);

  if (regen) {
    cudaMemcpy(d_dst, h_Src, imageW * imageH * sizeof(uchar4), cudaMemcpyHostToDevice);
    regen = false;
  }

  bool advance = (frame || running);
  if (advance) {
    cudaMemcpy(d_dst_old, d_dst, imageW * imageH * sizeof(uchar4), cudaMemcpyDeviceToDevice);
  }

  Poseidon_kernel(d_dst, d_dst_old, d_field, imageW, imageH, advance, pulse, px, py, fx, fy, init, rfact, tfact, fieldType);
  cudaThreadSynchronize();

  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

bool redraw = true;
void setRedraw(int value)
{
  redraw = true;
}

// OpenGL display function
void displayFunc(void)
{
  if (!running && !frame && !pulse) return;

  if (!redraw) return;
  redraw = false;
  glutTimerFunc(speed,&setRedraw,0);

  // render the Mandebrot image
  renderImage();

  init = false;
  frame = false;
  pulse = false;

  // load texture from PBO
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

  // fragment program is required to display floating point texture
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  glDisable(GL_DEPTH_TEST);

  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);

  glutSwapBuffers();
}

void cleanup()
{
  if (h_Src) {
    free(h_Src);
    h_Src = 0;
  }

  cudaGraphicsUnregisterResource(cuda_pbo_resource);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  glDeleteBuffers(1, &gl_PBO);
  glDeleteTextures(1, &gl_Tex);
  glDeleteProgramsARB(1, &gl_Shader);
}

void initMenus();

// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int)
{
  printf("k = %u\n",k);

  switch (k) {
    case '\033':
      printf("Shutting down...\n");
      exit(EXIT_SUCCESS);
      break;
    case ' ':
      printf("Space.\n");
      running = !running;
      break;
    case 'f':
      printf("Frame.\n");
      if (!running) frame = true;
      break;
    case 'q':
      rfact *= 1.2;
      break;
    case 'w':
      rfact /= 1.2;
      break;
    case 'a':
      tfact *= 1.01;
      break;
    case 's':
      tfact /= 1.01;
      break;
    case 'z':
      speed *= 1.2;
      break;
    case 'x':
      speed /= 1.2;
      break;
    case 't':
      fieldType = !fieldType;
      init = true;
      break;
    default:
      break;
  }

  printf("rfact = %f, tfact = %f, speed = %f\n",rfact,tfact,speed);
}

// OpenGL mouse click function
void clickFunc(int button, int state, int x, int y)
{
  if (!leftClicked) {
    printf("button = %i, state = %i, x = %i, y = %i\n",button,state,x,y);
  }

  if (button == 0)
    leftClicked = !leftClicked;
  if (button == 1)
    middleClicked = !middleClicked;
  if (button == 2)
    rightClicked = !rightClicked;

  if ((button == 1) && (!leftClicked)) {
    pulse = true;
    px = x;
    py = imageH-y;
  }

  if (button == 0) {
    fx = x;
    fy = imageH-y;
    init = true;
  }

  int modifiers = glutGetModifiers();
  if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT)) {
    leftClicked = 0;
    middleClicked = 1;
  }

  if (state == GLUT_UP) {
    leftClicked = 0;
    middleClicked = 0;
  }
}

// OpenGL mouse motion function
void motionFunc(int x, int y)
{
  if (leftClicked) {
    fx = x;
    fy = imageH-y;
    init = true;
  }
}

void idleFunc()
{
  glutPostRedisplay();
}

void mainMenu(int i)
{
}

void initMenus()
{
  glutCreateMenu(mainMenu);
  glutAddMenuEntry("Hardware double precision", 0);
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

// gl_Shader for displaying floating-point texture
static const char *shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
  GLuint program_id;
  glGenProgramsARB(1, &program_id);
  glBindProgramARB(program_type, program_id);
  glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

  GLint error_pos;
  glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
  if (error_pos != -1) {
    const GLubyte *error_string;
    error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
    fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
    return 0;
  }

  return program_id;
}

void initOpenGLBuffers(int w, int h)
{
  // delete old buffers
  if (h_Src) {
    free(h_Src);
    h_Src = 0;
  }

  if (d_dst_old) {
    cudaFree(d_dst_old);
    d_dst_old = 0;
  }

  if (d_field) {
    cudaFree(d_field);
    d_field = 0;
  }

  if (gl_Tex) {
    glDeleteTextures(1, &gl_Tex);
    gl_Tex = 0;
  }

  if (gl_PBO) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &gl_PBO);
    gl_PBO = 0;
  }

  // check for minimized window
  if ((w==0) && (h==0)) {
    return;
  }

  // allocate new buffers
  int size = w*h*4;
  h_Src = (uchar4*)malloc(size);
  cudaMalloc((void**)&d_dst_old,size);
  cudaMalloc((void**)&d_field,size*4);

  printf("Creating GL texture...\n");
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &gl_Tex);
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
  printf("Texture created.\n");

  printf("Creating PBO...\n");
  glGenBuffers(1, &gl_PBO);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);

  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard);
  printf("PBO created.\n");

  // load shader program
  gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void genRandImage()
{
  int size = imageW*imageH;
  int* isrc = (int*)h_Src;
  int cpos,r;
  for (int pos = 0; pos < size; pos++) {
    cpos = pos*4;
    r = rand();
    isrc[pos] = r;
    //h_Src[pos].x = r;
    //h_Src[pos].y = r;
    //h_Src[pos].z = r;
    h_Src[pos].w = 0;
  }
}

void reshapeFunc(int w, int h)
{
  glViewport(0, 0, w, h);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

  initOpenGLBuffers(w, h);
  imageW = w;
  imageH = h;

  fx = imageW/2;
  fy = imageH/2;

  regen = true;
  init = true;

  genRandImage();
}

void initGL(int argc, char **argv)
{
  printf("Initializing GLUT...\n");
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(imageW, imageH);
  glutInitWindowPosition(50, 50);
  glutCreateWindow(argv[0]);

  printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
  if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
    fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
    fprintf(stderr, "This sample requires:\n");
    fprintf(stderr, "  OpenGL version 1.5\n");
    fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
    fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
    exit(EXIT_FAILURE);
  }
  printf("OpenGL window created.\n");
}

void initData(int argc, char **argv)
{
  colors.w = 0;
  colors.x = 3;
  colors.y = 5;
  colors.z = 7;
  printf("Data initialization done.\n");
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  // parse command line arguments
  if (argc == 3) {
    tfact = atof(argv[1]);
    rfact = atof(argv[2]);
  }

  // Initialize OpenGL context first before the CUDA context is created.  This is needed
  // to achieve optimal performance with OpenGL/CUDA interop.
  initGL(argc, argv);
  initData(argc, argv);

  glutDisplayFunc(displayFunc);
  glutIdleFunc(idleFunc);
  glutKeyboardFunc(keyboardFunc);
  glutMouseFunc(clickFunc);
  glutMotionFunc(motionFunc);
  glutReshapeFunc(reshapeFunc);
  initMenus();

  atexit(cleanup);

  glutMainLoop();
  cudaThreadExit();
  exit(EXIT_SUCCESS);
}

