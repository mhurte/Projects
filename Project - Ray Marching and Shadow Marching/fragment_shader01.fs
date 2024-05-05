#version 330 core

uniform vec2 mousePos;      //[-1, 1]²
uniform float time;         //s
uniform float aspectRatio;  //width/heigth

in vec2 fragCoord; //[-1, 1]² position of pixel

out vec4 outColor;

#define DIST_MIN 1. // minimum distance to objects
#define DIST_MAX 70. // maximum distance to objects 

#define NB_STEP 2.

#define RAY_MARCH_STEPS 100
#define RAY_MARCH_PRECI 0.001
#define PI 3.14159265359


#define colors true // whether to plot normals (true) or z-buffer (false)

struct Material
{
  vec3 K_a; // ambiant
  vec3 K_d; // diffuse
  vec3 K_s; // specular
  float q;  // specular exponent
};

// ray structure
struct Ray 
{
  vec3 ro; // origin
  vec3 rd; // direction
};

struct Surface {
    vec3 p;
    float t; // surface distance
    vec3 c; // surface color
    Material mat;

};

// plane structure
struct Plane
{
  vec3 n; // normal
  float d; // offset
  Material mat;
};

struct ID
{
  float id;
  Material mat;
};

struct Container
{ 
  
  float m;
  Material mat;
  ID id;
};
struct Torus 
{
vec3 cen;
float lr;
float sr;
Material mat;
  
};
// sphere structure
struct Sphere 
{
  vec3 cen; // center
  float r; // radius
  vec3 c;
  Material mat;
};

struct Light
{ 
  vec3 pos; // position
  float I; // intensity
};




const Material m0 = Material(vec3(0.5,0.5,0.6),vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),1);
// Distance of the camera to the origin
const float DP = 35.0;

// Max distance for z-buffer "normalization"
const float DIST_DEPTH = 20.0;

// Sky color
vec3 background_color = vec3(0.52,0.8,0.918);

float sdSphere( vec3 p, float s ) {
  return length(p)-s;
}







float fPlane(vec3 p, vec3 n, float distanceFromOrigin) {
	return dot(p, n) + distanceFromOrigin;
}

//Function min that interpolates to give smoother transitions
float smin( float a, float b, float k ) {
    float h = clamp( 0.5+0.5*(b-a)/k, 0., 1. );
    return mix( b, a, h ) - k*h*(1.0-h);
}

//Dist to a Torus

float fTorus(vec3 p, float smallRadius, float largeRadius) {
	return length(vec2(length(p.xz) - largeRadius, p.y)) - smallRadius;
}


//Min between a sphere and a sphere returns the min and material associated in a Container struct
Container ssmmin(vec3 p,Sphere s1,Sphere s2,ID id1,ID id2) {
  if (sdSphere(s1.cen, s1.r) < sdSphere(s2.cen,s2.r)){
    return Container(sdSphere(s1.cen,s1.r),s1.mat,id1);
  }
  if ((sdSphere(s1.cen, s1.r) >= sdSphere(s2.cen,s2.r))){
    return Container(sdSphere(s2.cen,s2.r),s2.mat,id2);
  }
}
//Min between sphere and plane

Container spmmin(vec3 p,Sphere s1,Plane pl,ID id1,ID id2) {
  if (sdSphere(s1.cen, s1.r) < fPlane(p,pl.n,pl.d)){
    return Container(sdSphere(s1.cen,s1.r),pl.mat,id1);
  }
  if ((sdSphere(s1.cen, s1.r) >= fPlane(p,pl.n,pl.d))){
    return Container(fPlane(p,pl.n,pl.d),pl.mat,id2);
  }
}
//Min between sphere and Torus
Container stmmin(vec3 p,Sphere s1,Torus t1,ID id1,ID id2) {
  if (sdSphere(s1.cen, s1.r) < fTorus(p,t1.sr,t1.lr)){
    return Container(sdSphere(s1.cen,s1.r),s1.mat,id1);
  }
  if ((sdSphere(s1.cen, s1.r) >= fTorus(p,t1.sr,t1.lr))){
    return Container(fTorus(p,t1.sr,t1.lr),t1.mat,id2);
  }
}
//Min between plan and Torus
Container ptmmin(vec3 p,Plane pl, Torus t1, ID id1, ID id2){
  if (fPlane(p,pl.n,pl.d) < fTorus(p,t1.sr,t1.lr)){
    return Container(fPlane(p,pl.n,pl.d),pl.mat,id1);
  }
  if (fPlane(p,pl.n,pl.d) >= fTorus(p,t1.lr,t1.sr)){
    return Container(fTorus(p,t1.lr,t1.sr),t1.mat,id2);
  }
}

Surface scene(in vec3 p) {
    //Trunk
    Material m1 = Material(vec3(0.212,0.127,0.054),vec3(0.714,0.428,0.181),vec3(0.393,0.271,0.166),1);
    vec3 pos1 = vec3(p.x,p.y,p.z);
    ID id1 = ID(1.0,m1);
    pos1.y=pos1.y/7.;
    Sphere s1 = Sphere(pos1,1,vec3(0.5,1.,0.),m1);
    float d1 = 0.; 

    //Bee
    vec3 pos2 = (vec3(p.x,p.y,p.z));
    Material m2 = Material(vec3(0.1,0.1,0.1),vec3(0.5,0.5,0.),vec3(0.6,0.6,0.5),0.25);

    ID id2 = ID(2.0,m2);
    
    pos2.x= (pos2.x+10*cos(time/1.9))/1.1;
    pos2.y-=3+1.7*cos(1.5*time+cos(time));
    pos2.z= pos2.z+11*sin(time/2);
    Sphere s2 = Sphere(pos2,0.15,vec3(0.5,0.5,0.5),m2);
    float d2 = 0.;

    Container result1 = ssmmin(p,s1,s2,id1,id2);
    float result = smin(sdSphere(pos1,s1.r)+d1/2,sdSphere(pos2,s2.r)+d2/2,1);

    //Leaves
    Material m3 = Material(vec3(0.,0.2,0.),vec3(0.1,0.55,0.1),vec3(0.25,0.35,0.25),0.7);
    vec3 pos3 = (vec3(p.x,p.y,p.z));
    vec3 leaves = (pos3*6)+time;
    float d3=cos(leaves.x/4)*cos(leaves.y/2)*cos(leaves.z/4)/4;
    pos3.y-=8;
    pos3.x/=1.+cos(time/2+pos3.x/2.)/5.+sin(5*time)/200;
    pos3.y/=1.+cos(time/2+pos3.x/2.)/5.+sin(5*time)/200;
    pos3.z/=1.+cos(time/2+(pos3.x+pos3.z)/4.)/5.;
    ID id3 = ID(3.0,m3);


    Sphere s3 = Sphere(pos3,3,vec3(0.5,0.5,0.5),m3);

    if(result1.id.id==1.0){
      result1 = ssmmin(p,s1,s3,id1,id3);
      result = smin(sdSphere(pos1,s1.r)+d1/2,sdSphere(pos3,s3.r)+d3/2,0.8);
    }

    if(result1.id.id==2.0){
      result1 = ssmmin(p,s2,s3,id2,id3);
      result = smin(sdSphere(pos2,s2.r)+d2/2,sdSphere(pos3,s3.r)+d3/2,0.8);
    }


    //Grass
    Material m4 = Material(vec3(0.,0.2,0.),vec3(0.1,0.35+cos(p.x/2.+cos(time))/10+cos(p.z/3+cos(time)/2)/20.+cos(time)/20,0.1+cos(time)/20.),vec3(0.,0.15,0.),2);
    float d4= 0.;
    vec3 n1 = vec3(0,1,0);
    ID id4 = ID(4.0,m4);
    vec3 pos4 = p;
    pos4.y=pos4.y+abs(cos(0.8*pos4.x+time/2)+2*sin(0.8*pos4.z-time/2))/12;
    Plane pl = Plane(n1,d4,m4);


    
    if (result1.id.id==1.0){
      result1 = spmmin(pos4,s1,pl,id1,id4);
      result = smin(result,fPlane(pos4,n1,0),1);
    }

    if (result1.id.id==2.0){
      result1 = spmmin(pos4,s2,pl,id2,id4);
      result = smin(result,fPlane(pos4,n1,0),1);
    }

    if (result1.id.id==3.0){
      result1 = spmmin(pos4,s3,pl,id3,id4);
      result = smin(result,fPlane(pos4,n1,0),1);
    }

    //Fence
    vec3 p5 = p;
    p5.y-=1.8;
    p5.y/=2.0;
    Material m5 = Material(vec3(0.212,0.127,0.054),vec3(0.614-(abs(cos(p5.x*2))/5.+abs(sin(p5.z*2)/5.)),0.328-(abs(cos(p5.x*2))/7.+abs(sin(p5.z*2)/7.)),0.181),vec3(0.293,0.171,0.066),5);
    Torus t1 = Torus(p5,30,0.8,m5);
    float dtsurface = fTorus(p5,t1.sr,t1.lr);
    ID id5 = ID(5.0,m5);
  
    if (result1.id.id==1.0){
      result1 = stmmin(p,s1,t1,id1,id5);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==2.0){
      result1 = stmmin(p,s2,t1,id2,id5);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==3.0){
      result1 = stmmin(p,s3,t1,id3,id5);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==4.0){
      result1 = ptmmin(p,pl,t1,id4,id5);
      result = smin(result,dtsurface,0.5);
    }

//Cloud
    vec3 pos6 = (vec3(p.x,p.y,p.z));
    Material m6 = Material(vec3(0.5,0.5,0.5),vec3(0.85,0.85,0.85),vec3(0.7,0.7,0.7),1.);

    ID id6 = ID(6.0,m6);
    
    
    pos6.x= (pos6.x+10*cos(time/10))/2;
    pos6.y-=10;
    pos6.z= (pos6.z+10*sin(time/10))/1.2;
    vec3 cloud = (pos6*6)+time;
    Sphere s6 = Sphere(pos6,2,vec3(0.5,0.5,0.5),m6);

    dtsurface = sdSphere(s6.cen,s6.r)+(cos(cloud.x)*cos(1/2*cloud.y)*cos(cloud.z/2))/10;



    if (result1.id.id==1.0){
      result1 = ssmmin(p,s1,s6,id1,id6);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==2.0){
      result1 = ssmmin(p,s2,s6,id2,id6);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==3.0){
      result1 = ssmmin(p,s3,s6,id3,id6);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==4.0){
      result1 = spmmin(p,s6,pl,id6,id4);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==5.0){
      result1 = stmmin(p,s6,t1,id6,id5);
      result = smin(result,dtsurface,1);
    }


    //Bee2
    vec3 pos7 = (vec3(p.x,p.y,p.z));
    Material m7 = Material(vec3(0.1,0.1,0.1),vec3(0.5,0.5,0.),vec3(0.6,0.6,0.5),0.25);

    ID id7 = ID(7.0,m7);
    
    pos7.x= ((pos7.x-10*cos(time/1.9))/1.1)+1.;
    pos7.y-=3+1.7*cos(1.5*time+cos(time));
    pos7.z= (pos7.z-11*sin(time/2.1)-1.);
    Sphere s7 = Sphere(pos7,0.15,vec3(0.5,0.5,0.5),m7);
    dtsurface = sdSphere(s7.cen,s7.r);

    if (result1.id.id==1.0){
      result1 = ssmmin(p,s7,s1,id7,id1);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==2.0){
      result1 = ssmmin(p,s7,s2,id7,id2);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==3.0){
      result1 = ssmmin(p,s7,s3,id7,id3);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==4.0){
      result1 = spmmin(p,s7,pl,id7,id4);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==5.0){
      result1 = stmmin(p,s7,t1,id7,id5);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==6.0){
      result1 = ssmmin(p,s7,s6,id7,id6);
      result = smin(result,dtsurface,1);
    }


//Cloud
    vec3 pos8 = (vec3(p.x,p.y,p.z));
    Material m8 = Material(vec3(0.5,0.5,0.5),vec3(0.85,0.85,0.85),vec3(0.7,0.7,0.7),1.);

    ID id8 = ID(8.0,m8);
    
    
    pos8.x= (pos8.x+10*cos((7*PI/6.)+time/10.))/2.+1.;
    pos8.y-=11;
    pos8.z= (pos8.z+10*sin((7*PI/6.)+time/10))/1.2+1.;
    cloud = (pos8*6)+time;
    Sphere s8 = Sphere(pos8,1.8,vec3(0.5,0.5,0.5),m8);

    dtsurface = sdSphere(s8.cen,s8.r)+(cos(cloud.x)*cos(1/2*cloud.y)*cos(cloud.z/2))/10;



    if (result1.id.id==1.0){
      result1 = ssmmin(p,s8,s1,id8,id1);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==2.0){
      result1 = ssmmin(p,s8,s2,id8,id2);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==3.0){
      result1 = ssmmin(p,s8,s3,id8,id3);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==4.0){
      result1 = spmmin(p,s8,pl,id8,id4);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==5.0){
      result1 = stmmin(p,s8,t1,id8,id5);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==6.0){
      result1 = ssmmin(p,s8,s6,id8,id6);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==7.0){
      result1 = ssmmin(p,s8,s7,id8,id7);
      result = smin(result,dtsurface,1);
    }


//Rock
    vec3 pos9 = (vec3(p.x,p.y,p.z));
    Material m9 = Material(vec3(0.1,0.1,0.1),vec3(0.4,0.4,0.4),vec3(0.1,0.1,0.1),1);

    ID id9 = ID(9.0,m9);
    
    
    pos9.x= (pos9.x+4.)/2.1;
    pos9.y-=0.6;
    pos9.z= (pos9.z+1.2)/1.6;
    vec3 Rock = (pos9*4);
    Sphere s9 = Sphere(pos9,1.2,vec3(0.5,0.5,0.5),m9);

    dtsurface = sdSphere(s9.cen,s9.r)+cos(Rock.x)*cos(Rock.y/2)*cos(Rock.z/3)/10;



    if (result1.id.id==1.0){
      result1 = ssmmin(p,s9,s1,id9,id1);
      result = smin(result,dtsurface,0.1);
    }

    if (result1.id.id==2.0){
      result1 = ssmmin(p,s9,s2,id9,id2);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==3.0){
      result1 = ssmmin(p,s9,s3,id9,id3);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==4.0){
      result1 = spmmin(p,s9,pl,id9,id4);
      result = smin(result,dtsurface,0.05);
    }
    if (result1.id.id==5.0){
      result1 = stmmin(p,s9,t1,id9,id5);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==6.0){
      result1 = ssmmin(p,s9,s6,id9,id6);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==7.0){
      result1 = ssmmin(p,s9,s7,id9,id7);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==8.0){
      result1 = ssmmin(p,s9,s8,id9,id8);
      result = smin(result,dtsurface,1);
    }

//Flower 1/2
    vec3 pos10 = (vec3(p.x,p.y,p.z));
    Material m10 = Material(vec3(0.1+abs(cos(pos10.x)/5.+cos(pos10.y)/10.),0.1,0.1),vec3(1.,0.2,0.2),vec3(0.1,0.1,0.1),1);

    ID id10 = ID(10.0,m10);
    
    
    pos10.x= (pos10.x-2.)/3;
    pos10.y=(pos10.y-1.5);
    pos10.z= (pos10.z-2.)/3;
    vec3  Flower= (pos10*4);
    Sphere s10 = Sphere(pos10,0.2,vec3(0.5,0.5,0.5),m10);

    dtsurface = sdSphere(s10.cen,s10.r)+cos(Flower.x)*cos(Flower.y/2)*cos(Flower.z/3)/10;


 
    if (result1.id.id==1.0){
      result1 = ssmmin(p,s10,s1,id10,id1);
      result = smin(result,dtsurface,0.1);
    }

    if (result1.id.id==2.0){
      result1 = ssmmin(p,s10,s2,id10,id2);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==3.0){
      result1 = ssmmin(p,s10,s3,id10,id3);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==4.0){
      result1 = spmmin(p,s10,pl,id10,id4);
      result = smin(result,dtsurface,0.05);
    }
    if (result1.id.id==5.0){
      result1 = stmmin(p,s10,t1,id10,id5);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==6.0){
      result1 = ssmmin(p,s10,s6,id10,id6);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==7.0){
      result1 = ssmmin(p,s10,s7,id10,id7);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==8.0){
      result1 = ssmmin(p,s10,s8,id10,id8);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==9.0){
      result1 = ssmmin(p,s10,s9,id10,id9);
      result = smin(result,dtsurface,1);
    }


//Flower 2/2
    vec3 pos11 = (vec3(p.x,p.y,p.z));
    Material m11 = Material(vec3(0.,0.1,0.),vec3(0.1,0.55,0.1),vec3(0.15,0.25,0.15),0.8);

    ID id11 = ID(11.0,m11);
    
    
    pos11.x= (pos11.x-2.)*6;
    pos11.y-=(pos11.y+20)/(100.);
    pos11.z= (pos11.z-2.)*6;
    Sphere s11 = Sphere(pos11,0.8,vec3(0.5,0.5,0.5),m11);

    dtsurface = sdSphere(s11.cen,s11.r);


 
    if (result1.id.id==1.0){
      result1 = ssmmin(p,s11,s1,id11,id1);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==2.0){
      result1 = ssmmin(p,s11,s2,id11,id2);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==3.0){
      result1 = ssmmin(p,s11,s3,id11,id3);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==4.0){
      result1 = spmmin(p,s11,pl,id11,id4);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==5.0){
      result1 = stmmin(p,s11,t1,id11,id5);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==6.0){
      result1 = ssmmin(p,s11,s6,id11,id6);
      result = smin(result,dtsurface,1);
    }

    if (result1.id.id==7.0){
      result1 = ssmmin(p,s11,s7,id11,id7);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==8.0){
      result1 = ssmmin(p,s11,s8,id11,id8);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==9.0){
      result1 = ssmmin(p,s11,s9,id11,id9);
      result = smin(result,dtsurface,1);
    }
    if (result1.id.id==10.0){
      result1 = ssmmin(p,s11,s10,id11,id10);
      result = smin(result,dtsurface,1);
    }



    return Surface(p,result,s1.c,result1.id.mat);



}

Surface march(in Ray r) {
    float t = DIST_MIN;

    for(int i=0;i<RAY_MARCH_STEPS,t<=DIST_MAX;++i) {
        Surface s = scene(r.ro+t*r.rd);

        if(s.t<RAY_MARCH_PRECI) {
            return Surface(r.ro+t*r.rd,t+s.t,s.c,s.mat);
        }

        t = t+s.t;
    }

    return Surface(vec3(0,0,0),DIST_MAX,vec3(0.5,0.8,0.9),m0);
}

vec3 normalAt(in Surface s,in Ray r) {
    const float e = 0.01;
    vec3 p = r.ro+s.t*r.rd;
    float nx = scene(vec3(p.x+e,p.y,p.z)).t-scene(vec3(p.x-e,p.y,p.z)).t;
    float ny = scene(vec3(p.x,p.y+e,p.z)).t-scene(vec3(p.x,p.y-e,p.z)).t;
    float nz = scene(vec3(p.x,p.y,p.z+e)).t-scene(vec3(p.x,p.y,p.z-e)).t;

    return normalize(vec3(nx,ny,nz));
}

Ray camRay(in vec2 p) {
    // p is the current pixel coord, in [-1,1]

    // normalized mouse position
    vec2 m = mousePos.xy;
    
    // camera position 
    float d = DP;
    vec3 ro = DP * vec3(cos(m.x)*cos(m.y), -sin(m.y), -sin(m.x)*cos(m.y));
    //vec3 ro = vec3(d*(sin(6.0*m.x)),DP/1.5,d*(cos(6.0*m.x)));


    // target point
    vec3 ta = vec3(0.0,0.0,0.0);

    // camera view vector
    vec3 cw = normalize(ta-ro);

    // camera up vector
    vec3 cp = vec3(0.0,1.0,0.0);

    // camera right vector
    vec3 cu = normalize(cross(cw,cp));

    // camera (normalized) up vector
    vec3 cv = normalize(cross(cu,cw));
    
    float fovDeg = 45.;
    float fovRad = (fovDeg/360.)*2.*PI;
    float zf = 1./tan(fovRad/2.);
    
    // view vector, including perspective 
    vec3 rd = normalize(p.x*cu + p.y*cv*(aspectRatio) + 2.*cw);

    return Ray(ro,rd);
}

vec3 shade(in Surface s, in Ray r) {
    vec3 n = normalAt(s,r);
    vec3 v = r.rd;
    vec3 lightpos = vec3(0.0,50.0,3.0);
    vec3 l = normalize(lightpos-s.p);
    vec3 e = reflect(l,n);
    float diff = dot(n,l)*0.5+0.5;
    float spec = max(dot(e,v),0.);
    
    //shadows
    Ray r2 = Ray(s.p+n*0.02,normalize(l));
    float d = march(r2).t;
    if(d<length(l-s.p)){
      diff=0;
    }
    //light 
    vec3 kd = s.mat.K_d;
    vec3 ks = s.mat.K_s;
    float sh = s.mat.q;
    
    return s.mat.K_a+diff*kd + pow(spec,sh)*ks;
}




void main() 
{

  vec2 uv = fragCoord;


  Ray r = camRay(uv);
  Surface s = march(r);

  vec3 c = s.c;

  if(s.t<DIST_MAX) {
      c = shade(s,r);
  }

  outColor = vec4(c,1.0);
}
