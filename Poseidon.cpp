#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cstdio>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "Poseidon_kernel.h"

// openGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

// source image on the host side
uchar4 *h_screen = 0;

// destination image on the GPU side
uchar4 *d_screen = NULL;
uchar4 *d_screen_old = NULL;
uchar4 *d_cells = NULL;
float4 *d_field = NULL;

// original image width and height
int imageW = 800;
int imageH = 800;

// user interface variables
bool leftClicked = false;
bool middleClicked = false;
bool rightClicked = false;

bool running = true;
bool frame = false;
bool pulse = false;
bool calc_field = true;
bool draw_field = false;
bool regen = true;

// pulse and field positions
int px = 0;
int py = 0;
int fx = 0;
int fy = 0;

// timer ID
float speed = 10.0;

// default parameters
float tfact = 0.92;
float rfact = 1.0;
float width = 0.5;
float steep = 3.0;
int fieldType = 0;

/*
// karam's favorite
float rfact = 0.833333;
float tfact = 3.560367;
float width = 9.244208;
float steep = 0.135220;
int fieldType = 1;
*/

// unknown
#define CHAR_MASK 0x000000ff
#define BUFFER_DATA(i) ((char *)0 + i)
#define INT_TO_UCHAR4(r) (*((uchar4*)(&r)))
#define INT_TO_UCHAR(r) ((unsigned char)(r & CHAR_MASK))

// image types
#define IMAGE_TYPE_CUTOFF 0
#define IMAGE_TYPE_GAUSSIAN 1
#define IMAGE_TYPE_UNIFORM 2

void renderImage()
{
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_screen, &num_bytes, cuda_pbo_resource);

    if (regen) {
        cudaMemcpy(d_screen, h_screen, imageW * imageH * sizeof(uchar4), cudaMemcpyHostToDevice);
        regen = false;
    }

    bool advance = (frame || running);

    cudaMemcpy(d_screen_old, d_screen, imageW * imageH * sizeof(uchar4), cudaMemcpyDeviceToDevice);
    Poseidon_kernel(d_screen, d_screen_old, d_field, imageW, imageH, advance, pulse,
        px, py, fx, fy, rfact, tfact, width, steep, fieldType, draw_field, calc_field);
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
    if (!running && !frame && !pulse && !draw_field) return;

    if (!redraw) return;
    redraw = false;
    glutTimerFunc(speed,&setRedraw,0);

    // render the Mandebrot image
    renderImage();

    calc_field = false;
    draw_field = false;
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
    if (h_screen) {
        free(h_screen);
        h_screen = 0;
    }

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);
}

void initMenus();

inline float distance2(float x1, float y1, float x2, float y2)
{
    return sqrt(powf(x1-x2,2)+powf(y1-y2,2));
}

inline float gaussian(float x, float y, float cx, float cy, float sigma)
{
    float gx = exp(-0.5*powf((x-cx)/sigma,2));
    float gy = exp(-0.5*powf((y-cy)/sigma,2));
    return gx*gy;
}

void genRandImage(int type, bool indep)
{
    int ix, iy, pos;
    float cx, cy;
    int r0;
    float v;
    unsigned char c;
    uchar4 r;

    for (iy = 0; iy < imageH; iy++) {
        for (ix = 0; ix < imageW; ix++) {
            pos = iy*imageW + ix;
            cx = float(ix)/imageW;
            cy = float(iy)/imageH;
            switch (type) {
            case IMAGE_TYPE_CUTOFF:
                if (distance2(cx,cy,0.5,0.5) < 0.1) {
                    r0 = rand();
                    r = INT_TO_UCHAR4(r0);
                    r.w = 0;
                    if (!indep) {
                        r.y = r.x;
                        r.z = r.x;
                    }
                } else {
                    r = {0, 0, 0, 0};
                }            
                break;
            case IMAGE_TYPE_GAUSSIAN:
                v = rand()*gaussian(cx,cy,0.5,0.5,0.1);
                r0 = (int)(CHAR_MASK*v);
                c = INT_TO_UCHAR(r0);
                r = {c, c, c, 0};
                break;
            case IMAGE_TYPE_UNIFORM:
                r.w = 0;
                v = powf(rand(), 0.25);
                r0 = (int)(CHAR_MASK*v);
                r.x = r0;
                v = powf(rand(), 0.25);
                r0 = (int)(CHAR_MASK*v);
                r.y = r0;
                v = powf(rand(), 0.25);
                r0 = (int)(CHAR_MASK*v);
                r.z = r0;
                break;
            }
            h_screen[pos] = r;
        }
    }
}

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
        printf("Pause\n");
        running = !running;
        break;
    case 'f':
        printf("Advance frame\n");
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
    case 'o':
        width *= 1.2;
        break;
    case 'p':
        width /= 1.2;
        break;
    case 'k':
        steep *= 1.2;
        break;
    case 'l':
        steep /= 1.2;
        break;
    case 't':
        fieldType = (fieldType + 1) % 4;
        calc_field = true;
        break;
    case 'y':
        draw_field = true;
        running = false;
        break;
    case 'r':
        printf("Random image\n");
        regen = true;
        calc_field = true;
        genRandImage(IMAGE_TYPE_UNIFORM, false);
        break;
    default:
        break;
    }

    printf("\n[rule]\nself_couple = %f\nfield_couple = %f\nfield_type = %i\nrule_width = %f\nrule_steep = %f\ngrid_size='%ix%i'\nspeed = %f\n\n",tfact,rfact,fieldType,width,steep,imageW,imageH,1.0/speed);
}

// OpenGL mouse click function
void clickFunc(int button, int state, int x, int y)
{
    int modifiers = glutGetModifiers();
    printf("button = %i, state = %i, modifiers = %i, x = %i, y = %i\n",button,state,modifiers,x,y);

    if (button == 0) {
        if (state == GLUT_DOWN) {
            leftClicked = true;
        } else if (state == GLUT_UP) {
            leftClicked = false;
        }
    }

    if (button == 1) {
        if (state == GLUT_DOWN) {
            middleClicked = true;
        } else if (state == GLUT_UP) {
            middleClicked = false;
        }
    }

    if ((button == 0) && (state == GLUT_DOWN)) {
        pulse = true;
        px = x;
        py = imageH - y;
    }

    if ((button == 1) && (state == GLUT_DOWN)) {
        fx = x;
        fy = imageH - y;
        calc_field = true;
    }
}

// OpenGL mouse motion function
void motionFunc(int x, int y)
{
    if (leftClicked) {
        pulse = true;
        px = x;
        py = imageH-y;
    }

    if (middleClicked) {
        fx = x;
        fy = imageH-y;
        calc_field = true;
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
    if (h_screen) {
        free(h_screen);
        h_screen = 0;
    }

    if (d_screen_old) {
        cudaFree(d_screen_old);
        d_screen_old = 0;
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
    h_screen = (uchar4*)malloc(size);
    cudaMalloc((void**)&d_screen_old,size);
    cudaMalloc((void**)&d_field,size*4);

    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_screen);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_screen, GL_STREAM_COPY);

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard);
    printf("PBO created.\n");

    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void reshapeFunc(int w, int h)
{
    printf("size = %i x %i\n", w, h);

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
    calc_field = true;

    genRandImage(IMAGE_TYPE_CUTOFF, true);
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
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // parse command line arguments
    int narg = argc - 1;
    if (narg >= 1) {
        tfact = atof(argv[1]);
    }
    if (narg >= 2) {
        rfact = atof(argv[2]);
    }
    if (narg >= 3) {
        width = atof(argv[3]);
    }
    if (narg >= 4) {
        steep = atof(argv[4]);
    }
    if (narg >= 5) {
        fieldType = atoi(argv[5]);
    }
    if (narg >= 6) {
        imageW = atoi(argv[6]);
    }
    if (narg >= 7) {
        imageH = atoi(argv[7]);
    }
    if (narg >= 8) {
        speed = atof(argv[8]);
    }

    printf("size = %i x %i\n", imageW, imageH);
    printf("tfact = %f\n", tfact);
    printf("rfact = %f\n", rfact);
    printf("width = %f\n", width);
    printf("steep = %f\n", steep);
    printf("field = %i\n", fieldType);
    printf("speed = %f\n", speed);

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
